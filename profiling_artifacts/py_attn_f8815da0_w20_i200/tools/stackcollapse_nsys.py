#! /usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

NSYS_CLI_BINARY_NAME = "nsys.exe" if os.name == "nt" else "nsys"

SELECT_CALLCHAIN_TABLE_EXISTENCE = """
SELECT
    count(name)
FROM
    sqlite_master 
WHERE 
    type='table' AND name = 'sampling_callchains'
COLLATE NOCASE
"""

SELECT_CALLSTACKS = """
WITH usage(id, cycles) AS 
(
    SELECT
        sc.id,
        SUM(cpucycles) AS cycles 
    FROM
        sampling_callchains sc 
        LEFT JOIN
            composite_events se 
            ON sc.id == se.id 
    WHERE
        sc.stackdepth == 0 
    GROUP BY
        sc.symbol,
        sc.module 
)
SELECT
    GROUP_CONCAT(value, ';') || ' ' || cycles 
FROM
    (
        SELECT
            si.value,
            sc.id,
            u.cycles 
        FROM
            usage u 
            INNER JOIN
                sampling_callchains sc 
                ON sc.id = u.id 
            INNER JOIN
                stringids AS si 
                ON sc.symbol == si.id 
            INNER JOIN
                stringids AS sm 
                ON sc.module == sm.id 
        WHERE
            si.value <> '[Max depth]' 
        ORDER BY
            stackdepth DESC 
    )
GROUP BY
    id 
ORDER BY
    cycles DESC
"""


REGEX_TYPE_MODIFIER = "(?:(?:(?:unsigned)|(?:signed)|(?:long)) *){0,2}"
REGEX_TYPE_SPECIFIER = "[ \*&]*(?:const)?[ \*&]*"
REGEX_IDENTIFIER = "(?:(?:[^\W\d]|~)(?:[^\W]|[<>\*&\[\]])*)"

REGEX_SPECIAL_IDENTIFIERS = "(?:\(anonymous namespace\))|(?:\{lambda\(\)#?\d*\})|(?:decltype ?\(\))"

REGEX_NON_SPEC_TYPE_IDENTIFIER = "(?:{type_modifier}{identifier})".format(
    type_modifier=REGEX_TYPE_MODIFIER, identifier=REGEX_IDENTIFIER
)
REGEX_QUALIFIED_NON_SPEC_TYPE_IDENTIFIER = (
    "{non_spec_type_id}{type_specifier}(?:::{non_spec_type_id}{type_specifier})*(?:(?:::)?)?".format(
        non_spec_type_id=REGEX_NON_SPEC_TYPE_IDENTIFIER, type_specifier=REGEX_TYPE_SPECIFIER
    )
)

OVERLOADED_OPERATORS_LIST = [
    "+",
    "-",
    "*",
    "/",
    "%",
    "^",
    "&",
    "|",
    "~",
    "!",
    "=",
    "<",
    ">",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "^=",
    "&=",
    "|=",
    "<<",
    ">>",
    "<<=",
    ">>=",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "++",
    "--",
    ",",
    "->*",
    "->",
    "()",
    "[",
    "]",
    "new",
    "delete",
    "new[]",
    "delete[]",
]

REGEX_OVERLOADED_OPERATORS_LIST = [re.escape(op) for op in OVERLOADED_OPERATORS_LIST]
REGEX_OPERATOR_OVERLOADS = "(?:" + ")|(?:".join(REGEX_OVERLOADED_OPERATORS_LIST) + ")"
REGEX_OPERATOR_IDENTIFIER = "(?:operator *(?:<>)?(?:{type_id}|(?:{operator_overloads}))?(?:<>)?)".format(
    type_id=REGEX_QUALIFIED_NON_SPEC_TYPE_IDENTIFIER, operator_overloads=REGEX_OPERATOR_OVERLOADS
)

REGEX_TYPE_IDENTIFIER = "(?:{type_modifier}{identifier}|{special_ids}|{operator_id})".format(
    type_modifier=REGEX_TYPE_MODIFIER,
    identifier=REGEX_IDENTIFIER,
    special_ids=REGEX_SPECIAL_IDENTIFIERS,
    operator_id=REGEX_OPERATOR_IDENTIFIER,
)
REGEX_QUALIFIED_TYPE_IDENTIFIER = "{type_id}{type_specifier}(?:::{type_id}{type_specifier})*(?:(?:::)?)?".format(
    type_id=REGEX_TYPE_IDENTIFIER, type_specifier=REGEX_TYPE_SPECIFIER
)

FUNCTION_NAME_DELIMITER = "(?:[ &\*>])"
FUNCTION_ARGUMENTS = "(?:\(\))?"

APPROXIMATE_FUNCTION_STRING = "^((?:(?:{type_specifier}{func_name_delimiter})?{qualified_type_id}(?:{func_name_delimiter}{type_specifier})?{func_name_delimiter}))?({qualified_type_id})(?:{func_arg}{type_specifier})?$".format(
    qualified_type_id=REGEX_QUALIFIED_TYPE_IDENTIFIER,
    func_name_delimiter=FUNCTION_NAME_DELIMITER,
    func_arg=FUNCTION_ARGUMENTS,
    type_specifier=REGEX_TYPE_SPECIFIER,
)
APPROXIMATE_FUNCTION_REGEX = re.compile(APPROXIMATE_FUNCTION_STRING, re.U)


def collapse_parentheses(str_, left_p, right_p):
    """Collapse everything between matching left_p and right_p (including collapsing of nested matches)

    Args:
        str_ (string): String to collapse.
        left_p (string): Left delimiter
        right_p (string): Right delimiter

    Returns:
        string: Collapsed string.
    """
    shortened_str = ""
    inner_value = ""
    parentheses_lvl = 0
    for c in str_:
        if parentheses_lvl == 0 or (parentheses_lvl == 1 and c == right_p):
            if c == right_p:
                if inner_value == "anonymous namespace":
                    shortened_str += inner_value
                inner_value = ""
            shortened_str += c

        if c == right_p:
            parentheses_lvl -= 1
        if parentheses_lvl == 1:
            inner_value += c
        if c == left_p:
            parentheses_lvl += 1
    return shortened_str


def shorten_function_name_approximately(full_function_def):
    """Try to shorten function name (in some cases shortening may fail and return the original filename).

    Args:
        full_function_def (string): Original filename.

    Returns:
        string: Shortened or full function name.
    """
    prepared_function_def = collapse_parentheses(full_function_def, "(", ")")
    prepared_function_def = collapse_parentheses(prepared_function_def, "<", ">")
    prepared_function_def = collapse_parentheses(prepared_function_def, "[", "]")
    prepared_function_def = re.sub(r"[\n\t\s]+", " ", prepared_function_def)
    prepared_function_def = re.sub(r"([ \*\&>\}\)\]])const::", r"\1::", prepared_function_def)
    prepared_function_def = re.sub(r"\(\) *::", "::", prepared_function_def)
    m = re.search(APPROXIMATE_FUNCTION_REGEX, prepared_function_def)
    # may be a function address or a complex function name
    if not m:
        return full_function_def

    function_name = ""
    if m.group(1) and m.group(1).find("operator") != -1 and m.group(2):
        function_name = m.group(1) + " " + m.group(2)
        function_name = re.sub(r"[\n\t\s]+", " ", function_name)
    elif m.group(2):
        function_name = m.group(2)
    else:
        function_name = full_function_def
    return function_name


def shorten_func_names_approximately(flamegraph_row, full_function_names):
    """Try to shorten function names (in some cases shortening may fail and return the original filenames).

    Args:
        flamegraph_row (string): String suitable for flamegraph with original function names
        full_function_names (string): Use full function names with return type, arguments and expanded templates, if available.

    Returns:
        list[string]: List of shortened or full function names.
    """
    if full_function_names:
        return flamegraph_row
    flamegraph_parts = flamegraph_row.split(" ")
    if len(flamegraph_parts) < 2:
        return flamegraph_row
    cycles_cnt = flamegraph_parts[-1]
    full_function_defs = " ".join(flamegraph_parts[:-1]).split(";")
    full_function_defs_wo_recursive = []
    if len(full_function_defs) > 1:
        prev_val = full_function_defs[0]
        for val in full_function_defs[1:]:
            if val != prev_val:
                full_function_defs_wo_recursive.append(val)
    else:
        full_function_defs_wo_recursive = full_function_defs

    flamegraph_row = ";".join(
        [
            shorten_function_name_approximately(full_function_def)
            for full_function_def in full_function_defs_wo_recursive
        ]
    )

    flamegraph_row += " {}".format(cycles_cnt)

    return flamegraph_row


def check_cpu_samples_exists(conn):
    """Check if CPU callstacks exist in SQLite database

    Args:
        conn: SQLite database connection

    Returns:
        bool: True if CPU callstacks exist in SQLite database
    """
    c = conn.cursor()
    c.execute(SELECT_CALLCHAIN_TABLE_EXISTENCE)
    if c.fetchone()[0] == 1:
        return True

    return False


def convert_to_collapsed(sqlite_db_path, outfile_name, full_function_names):
    """Convert CPU callstacks from a SQLite database file to an output suitable for flamegraph.pl

    Args:
        sqlite_db_path (string): _description_
        outfile_name (string): Path to a results file. If None or empty an output is written to stdout.
        full_function_names (bool): Use full function names with return type, arguments and expanded templates, if available.
    """
    with sqlite3.connect(sqlite_db_path) as conn:
        if not check_cpu_samples_exists(conn):
            sys.stderr.write("Report does not contain CPU samples. Folded output will not be generated.\n")
            return

        c = conn.cursor()
        c.execute(SELECT_CALLSTACKS)
        if outfile_name:
            with open(outfile_name, "w", encoding="utf-8") as outfile:
                for row in c:
                    outfile.write(shorten_func_names_approximately(row[0], full_function_names) + "\n")
        else:
            for row in c:
                print(shorten_func_names_approximately(row[0], full_function_names))


def export_to_sqlite(nsys_target_bin_path, nsys_rep_path, sqlite_db_path):
    """Export Nsight Systems report to a SQLite database file

    Args:
        nsys_target_bin_path (string): Path to a target Nsight Systems binary.
        nsys_rep_path (string): Path to a Nsight Systems report.
        sqlite_db_path (string): Path to a SQLite database file.

    Raises:
        ChildProcessError
    """
    popen = subprocess.Popen([nsys_target_bin_path, "export", "--type", "sqlite", "-o", sqlite_db_path, nsys_rep_path])
    stdout, stderr = popen.communicate()
    exit_code = popen.wait()
    if exit_code != 0:
        err_str = "Nsight Systems CLI export failed with exit code {}: {}\n{}".format(
            exit_code, (stdout or b"").decode("utf-8", "ignore"), (stderr or b"").decode("utf-8", "ignore")
        )
        print(err_str)
        raise ChildProcessError(err_str)


def get_arm_type_target_path_part(arch):
    """Get the folder name part identifying the current supported armv8 arch (SBSA or tegra)

    Args:
        arch (string): Current architecture

    Returns:
        string: "-sbsa", "-tegra" or ""
    """
    if arch != "armv8":
        return ""

    tegra_check_popen = subprocess.Popen(
        "find /proc/device-tree/ -maxdepth 1 -name 'tegra*' || echo ERROR",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tegra_check_output, _ = tegra_check_popen.communicate()
    if (tegra_check_output or b"").decode("utf-8", "ignore").startswith("/"):
        return "-tegra"

    sbsa_check_popen = subprocess.Popen(
        "lsmod | grep 'nvidia' || echo ERROR", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    sbsa_check_output, _ = sbsa_check_popen.communicate()
    if (sbsa_check_output or b"").decode("utf-8", "ignore").startswith("nvidia"):
        return "-sbsa"

    return ""


def get_target_bin_path(nsys_target_bin_path_arg):
    """Try to retrieve a Nsight Systems CLI path from argument (if it exists) or from the default installation path.
    Args:
        nsys_target_bin_path_arg (string): User path to the Nsight Systems CLI binary

    Raises:
        FileNotFoundError: Nsight Systems CLI not found

    Returns:
        string: Path to the Nsight Systems CLI binary.
    """

    if nsys_target_bin_path_arg:
        nsys_target_search_path = Path(nsys_target_bin_path_arg) / NSYS_CLI_BINARY_NAME
        if nsys_target_search_path.is_file():
            return str(nsys_target_search_path)

    nsys_host_path = Path(__file__).resolve().parent.parent.parent
    nsys_host_folder_name = nsys_host_path.name
    nsys_host_folder_name_parts = nsys_host_folder_name.split("-")
    nsys_target_path = None
    if len(nsys_host_folder_name_parts) == 3:
        host_os = nsys_host_folder_name_parts[1]
        # assume actual host architecture
        host_arch = nsys_host_folder_name_parts[2]
        nsys_target_folder_name = "target-" + host_os + get_arm_type_target_path_part(host_arch) + "-" + host_arch
        nsys_target_search_path = nsys_host_path.parent / nsys_target_folder_name / NSYS_CLI_BINARY_NAME
        if nsys_target_search_path.is_file():
            nsys_target_path = str(nsys_target_search_path)

    if not nsys_target_path:
        raise FileNotFoundError(
            "Nsight Systems CLI binary (nsys) not found."
            'Use "--nsys" argument to set an Nsight Systems CLI binary path.'
        )
    return nsys_target_path


def collapse_callstacks(nsys_target_bin_path, nsys_rep_path, outfile_name, full_function_names):
    """Export CPU callstacks from a Nsight Systems report file to a SQLite database and
    convert them to an output suitable for flamegraph.pl

    Args:
        nsys_target_bin_path (string): Path to a Nsight Systems CLI directory.
        nsys_rep_path (string): Path to a Nsight Systems report.
        outfile_name (string): Path to a results file. If None or empty an output is written to stdout.
        full_function_names (bool): Use full function names with return type, arguments and expanded templates, if available.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        sqlite_db_path = os.path.join(tmp_dir, "db.sqlite")
        export_to_sqlite(nsys_target_bin_path, nsys_rep_path, sqlite_db_path)
        convert_to_collapsed(sqlite_db_path, outfile_name, full_function_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for parsing Nsight Systems report files containing CPU call stacks and producing an output"
        "suitable for flamegraph.pl."
    )
    parser.add_argument("--nsys", action="store", help="Path to the Nsight Systems CLI directory", required=False)
    parser.add_argument(
        "-o",
        "--out",
        action="store",
        help="Path to the output file name (by default an output is written to stdout)",
    )
    parser.add_argument(
        "--full_function_names",
        default=False,
        action="store_true",
        help="Use full function names with return type, arguments and expanded "
        "templates, if available (default: false).",
    )
    parser.add_argument("nsys_rep_file", help="Nsight Systems report file path")
    args = parser.parse_args()

    custom_outfile_name = None
    if "out" in args and args.out:
        custom_outfile_name = args.out

    collapse_callstacks(
        get_target_bin_path(args.nsys), args.nsys_rep_file, custom_outfile_name, args.full_function_names
    )
