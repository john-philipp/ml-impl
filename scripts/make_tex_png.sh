#!/bin/bash

logfile=$(mktemp)
function _log() {
  echo $@ | tee -a ${logfile}
}

_log [INFO] Will write logs here: ${logfile} ...

name=${1:-log-reg}
dpi=${2:-100}
border=${3:-50}
timestamp=$(date +"%Y%m%d_%H%M%S")

src_dir=docs/tex
src_file=${src_dir}/${name}.tex

dest_dir=docs/images
dest_file=${dest_dir}/${name}-${timestamp}.png

if [[ ! -d docs ]]; then
  _log [ERROR] Run from root dir.
  exit 1
fi

if [[ ! -f ${src_file} ]]; then
  _log [ERROR] No such file: ${src_file}.
  exit 1
fi

set -e
rm -rf .tmp
mkdir -p .tmp docs/images
rm docs/images/${name}*.png || _log [INFO] Nothing to remove.
cd .tmp

_log [INFO] Generating dvi: name=${name} ...
# First run to generate references. Second to use them.
latex ../docs/tex/${name}.tex >> ${logfile}
latex ../docs/tex/${name}.tex >> ${logfile}

_log [INFO] Generating png: name=${name} dpi=${dpi} ...
dvipng -D ${dpi} ${name}.dvi -o ${name}.png >> ${logfile} 2>&1

_log [INFO] Running conversion: name=${name} border=${border} ...
convert ${name}.png -bordercolor white -border ${border}x${border} ../${dest_file} >> ${logfile}

cd ..
rm -rf .tmp

_log [INFO] Adding to index: ${dest_file} ...
git add ${dest_file}

_log [INFO] Updating readme ...
sed -i "s|${dest_dir}/${name}-.*\.png|${dest_file}|" readme.md

_log [INFO] Done.
_log [INFO] Logs written here: ${logfile} ...
