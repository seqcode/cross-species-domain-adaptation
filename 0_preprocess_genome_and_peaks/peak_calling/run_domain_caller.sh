#!/bin/bash

# This script was used to call domains for accessibility data.
# This was not used in the actual manuscript, but I'm keeping
# the script for posterity.


# give args $genome, $assay, and $cell

case "$assay" in
  ATAC*(-)*([Ss][Ee][Qq])) assay="ATACSeq";;
  DN[Aa][Ss][Ee]*(-)*([Ss][Ee][Qq])) assay="DNase";;
  *)
    echo "Error: accessibility type not correctly specified."
    exit 1
  ;;
esac
[ -z "$assay" ] && exit 1

case "$genome" in
  mm10) genome_long="Mus musculus;mm10";;
  hg38) genome_long="Homo sapiens;hg38";;
  *)
    echo "Error: genome not recognized."
    exit 1
  ;;
esac

folder="${genome}/${cell}/${assay}"


echo "Getting domain calls for ${assay}, ${cell}, ${genome_long} - ${genome}."

cd "/storage/home/kxc1032/group/lab/kelly/experimental_data/" || exit

echo "Running DomainFinder..."

java -Xmx30G org.seqcode.projects.seed.DomainFinder --loadpairs --threads 4 --species "${genome_long}" --seq "/storage/home/kxc1032/group/genomes/${genome}/" --design "${folder}/domains.design" --binwidth 50 --binstep 25 --mergewin 200 --poisslogpthres -5 --binpthres 0.05 --out "${folder}/domains" > "${folder}/domainFinder.out"  2>&1

echo "Done!"
