home_dir=$(dirname $0)
cd $home_dir

# echo remove older versions first
rm -rf GenePlexusZooManuscript.tar.gz
rm -rf data
rm -rf results
rm -rf figures


wget https://zenodo.org/record/10246207/files/GenePlexusZooManuscript.tar.gz
tar -xvzf "GenePlexusZooManuscript.tar.gz" --strip-components=1
rm -rf "GenePlexusZooManuscript.tar.gz"