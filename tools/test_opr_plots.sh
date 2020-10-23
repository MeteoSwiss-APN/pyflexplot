#!/bin/bash

# Script directory
DIR="$(dirname "$(readlink -f "$0")")"

# Check arguments
if [ ${#} -lt 2 ] || [ ${#} -gt 3 ]; then
    echo "error: wrong number of arguments (${#})" >&2
    echo "usage: $(basename "${0}") PRESET INFILE_PATH [WORK_DIR]" >&2
    exit 1
fi

# 1st argument: Preset name
preset="${1}"

# 2nd argument: Input file path
infile_path="${2}"
infile_path="$(readlink -f "${infile_path}")"

# 3rd argument (optional): Working directory (existing or not)
work_dir="${3:-test_preset_plots/${preset}}"

ref_dir="${work_dir}/ref"
test_dir="${work_dir}/test"
diff_dir="${work_dir}/diff"

echo
echo "work dir : ${work_dir}"
echo "ref dir  : ${ref_dir}"
echo "test dir : ${test_dir}"
echo "diff dir : ${diff_dir}"
echo

# Check the name of the local git repository
function check_git_repo()
{
    local name="${1}"
    if [ ! -d ".git" ]; then
        echo "error: not in a git repository" >&2
        return 1
    fi
    loc_name="$(basename "$(git config --get remote.origin.url)" | cut -d. -f1)"
    [ "${loc_name}" = "${name}" ] && return 0
    echo "error: wrong git repository: expecting '${name}' not '${loc_name}'" >&2
    return 1
}

# Check that a virtual environment is active
function check_active_venv()
{
    [ ! -z ${VIRTUAL_ENV} ] && return 0
    echo "error: no active virtual environment found" >&2
    return 1
}

comp_ref="true"
comp_test="true"

# Prepare directory for plots
function prepare_dest_dir()
{
    local path="${1}"
    local name="${2}"
    if [ ! -d "${path}" ]; then
        mkdir -pv "${path}"
    else
        # Check whether directory is empty
        n_tot="$(\ls "${path}" | wc -l)"
        if [ "${n_tot}" -gt 0 ]; then
            n_dirs="$(\ls -d "${path}"/*/ | wc -l)"
            n_files=$((n_tot - n_dirs))
            echo "found ${n_files} files and ${n_dirs} directories in ${path}"
            # Ask whether to delete contents, and if yes, do so
            echo
            while read -p "remove contents of ${name} directory (Y/n)? " reply; do
                case "${reply}" in
                    ''|[yY]|[yY][eE][sS]) \rm -rfv "${path}"/*;;
                    [nN]|[nN][oO]) ;;
                    *) echo "invalid reply '${reply}'" >&2; continue;;
                esac
                break
            done
            echo
        fi
    fi
}

# Get reference revision (tag/commit)
function get_ref_rev()
{
    local ref_rev="$(git describe --abbrev=0)"
    while read -p "reference tag/commit [${ref_rev}]? " reply; do
        [ "${reply}" = "" ] && break
        if ! git rev-parse "${reply}" >/dev/null 2>&1; then
            echo "invalid tag/commit '${reply}'" >&2
            continue
        fi
        ref_rev="${reply}"
        break
    done
    echo "${ref_rev}"
}

# Get current revision (branch/commit)
function get_curr_rev()
{
    local rev=$(git rev-parse --abbrev-ref HEAD)
    if [ "${rev}" = "HEAD" ] ; then
        # Detached head; obtain commit hash
        rev=$(git rev-parse HEAD)
    fi
    echo "${rev}"
}

# Handle existing reference directory
if [ -d "${ref_dir}" ]; then
    echo "reference directory '${ref_dir}' already exists"
    # Ask whether to recompute the reference plots
    echo
    while read -p "recompute reference plots (y/N)? " reply; do
        case "${reply}" in
            [yY]|[yY][eE][sS]) comp_ref="true";;
            ''|[nN]|[nN][oO]) comp_ref="false";;
            *) echo "invalid reply '${reply}'" >&2; continue;;
        esac
        break
    done
    echo
fi

# Prepare reference plots
if [ "${comp_ref}" = "true" ]; then
    echo "compute reference plots"
    check_git_repo "pyflexplot" || exit 1
    if ! check_active_venv; then
        echo -e "\nPlease activate your pyflexplot venv and try again!" >&2
        exit 1
    fi

    # Prepare reference directory
    prepare_dest_dir "${ref_dir}" "reference"

    # Ask for reference tag/commit (default: latest tag)
    ref_rev="$(get_ref_rev)"

    # Switch to reference tag/commit
    curr_rev="$(get_curr_rev)"
    echo -e "\nswitching from branch/commit ${curr_rev} to reference tag/commit '${ref_rev}'"
    echo "git checkout '${ref_rev}'"
    git -c advice.detachedHead=false checkout "${ref_rev}" || exit 1

    # Enter working directory
    curr_dir="${PWD}"
    echo -e "\nentering working directory"
    cd "${work_dir}" || exit 1
    pwd

    # Create plots
    ref_dir_rel="${ref_dir#${work_dir}/}"
    version=$(pyflexplot -V)
    echo -e "\ncomputing reference plots with pyflexplot ${version}:"
    # + cmd="pyflexplot -P8 --preset='${preset}' --setup infile '${infile_path}' --dest='${ref_dir_rel}' --no-show-version" || exit 1
    cmd="pyflexplot -P8 --preset='${preset}' --setup infile '${infile_path}' --dest-dir='${ref_dir_rel}'" || exit 1
    echo -e "${cmd}\n"
    eval "${cmd}"

    # Exit working directory
    echo -e "\nexiting working directory"
    cd "${curr_dir}" || exit 1
    pwd

    # Switch back to previous branch/commit
    echo -e "\nswitching back from reference tag/commit ${ref_rev} to branch/commit ${curr_rev}"
    echo "git checkout '${curr_rev}'"
    git checkout "${curr_rev}" || exit 1
fi

# Handle existing test directory
if [ -d "${test_dir}" ]; then
    echo "test directory '${ref_dir}' already exists"
    # Ask whether to recompute the test plots
    echo
    while read -p "recompute test plots (y/N)? " reply; do
        case "${reply}" in
            [yY]|[yY][eE][sS]) comp_test="true";;
            ''|[nN]|[nN][oO]) comp_test="false";;
            *) echo "invalid reply '${reply}'" >&2; continue;;
        esac
        break
    done
    echo
fi

# Prepare test plots
if [ "${comp_test}" = "true" ]; then
    echo "compute test plots"
    check_git_repo "pyflexplot" || exit 1
    if ! check_active_venv; then
        echo -e "\nPlease activate your pyflexplot venv and try again!" >&2
        exit 1
    fi

    # Prepare test directory
    prepare_dest_dir "${test_dir}" "test"

    # Enter working directory
    curr_dir="${PWD}"
    echo -e "\nentering working directory"
    cd "${work_dir}" || exit 1
    pwd

    # Create plots
    test_dir_rel="${test_dir#${work_dir}/}"
    version=$(pyflexplot -V)
    echo -e "\ncomputing test plots with pyflexplot ${version}:"
    # SR_TMP < TODO Change to --dest once old versions are no longer relevant
    # + cmd="pyflexplot -P8 --preset='${preset}' --setup infile '${infile_path}' --dest='${test_dir_rel}' --no-show-version" || exit 1
    cmd="pyflexplot -P8 --preset='${preset}' --setup infile '${infile_path}' --dest-dir='${test_dir_rel}'" || exit 1
    # SR_TMP >
    echo -e "${cmd}\n"
    eval "${cmd}"

    # Exit working directory
    echo -e "\nexiting working directory"
    cd "${curr_dir}" || exit 1
    pwd
fi

# Compare plots
prepare_dest_dir "${diff_dir}" "diffs"
echo -e "\ncomparing plots: test vs. reference\n"
for test_plot in "${test_dir}"/*; do
    plot_name="$(basename "${test_plot}")"
    ref_plot="${ref_dir}/${plot_name}"
    diff_plot="${diff_dir}/${plot_name}"

    # Check for identity with checksum
    md5_test=$(md5sum "${test_plot}" | cut -d' ' -f1)
    md5_ref=$(md5sum "${ref_plot}" | cut -d' ' -f1)
    if [ "${md5_test}" = "${md5_ref}" ]; then
        echo "[  OK  ] ${plot_name}"
        continue
    fi

    # Quantify difference
    compare "${test_plot}" "${ref_plot}" "${diff_plot}"

    echo "[ FAIL ] ${plot_name}"
    echo "         > ${diff_plot}"
done
