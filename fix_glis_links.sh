#!/usr/bin/env bash

OLD='https://ssl.fao.org/glis/doi/'
NEW='https://glis.fao.org/glis/doi/'

find . -type f -name "*Germplasm_DOIs*" -print0 |
while IFS= read -r -d '' file; do
    echo "Fixing: $file"
    sed -i "s#${OLD}#${NEW}#g" "$file"
done
