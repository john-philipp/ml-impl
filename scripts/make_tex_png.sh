#!/bin/bash
name=${1:-log-reg}
dpi=${2:-100}
border=${3:-50}
timestamp=$(date +"%Y%m%d_%H%M%S")
dest_dir=docs/images
dest_file=docs/images/${name}-${timestamp}.png

if [[ ! -d docs ]]; then
  echo [ERROR] Run from root dir.
  exit 1
fi

set -e
rm -rf .tmp
mkdir -p .tmp docs/images
rm docs/images/${name}*.png || echo [INFO] Nothing to remove.
cd .tmp

echo [INFO] Generating dvi: name=${name} ...
latex ../docs/tex/${name}.tex &> /dev/null

echo [INFO] Generating png: name=${name} dpi=${dpi} ...
dvipng -D ${dpi} ${name}.dvi -o ${name}.png &> /dev/null

echo [INFO] Running conversion: name=${name} border=${border} ...
convert ${name}.png -bordercolor white -border ${border}x${border} ../${dest_file} &> /dev/null

cd ..
rm -rf .tmp

echo [INFO] Adding to index: ${dest_file} ...
git add ${dest_file}

echo [INFO] Updating readme ...
sed -i "s|${dest_dir}/${name}-.*\.png|${dest_file}|" readme.md

echo [INFO] Done.