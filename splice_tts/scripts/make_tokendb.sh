set -e
set -o pipefail

log() {
    echo -e $*
}

# directories
ctm=local/ls_train_960/phn.ctm
exp=exp/

# configs
tokendb_min_ngram=3
tokendb_max_ngram=10
tokendb_min_candidates=-1
tokendb_keep_candidates=100

export PYTHONPATH=$PWD/../../:$PYTHONPATH
. utils/parse_options.sh

tokendb_dir=$exp/tokendb
mkdir -p ${tokendb_dir}/logs
trap 'kill $(jobs -p)' EXIT
for ngram in `seq ${tokendb_min_ngram} ${tokendb_max_ngram}`; do
    min_candidates=${tokendb_min_candidates}
    if [ ${tokendb_min_candidates} -eq -1 ]; then
        min_candidates=$((3 > (10 - ngram) ? 3 : (10 - ngram)))
    fi
    log "generating token stats for ${ngram} token"
    python -m splice_tts.bin.ctm2tokendb --ctm $ctm --ngram ${ngram} --output ${tokendb_dir}/${ngram}.shelve --min_candidates ${min_candidates} --keep_candidates ${tokendb_keep_candidates} > ${tokendb_dir}/logs/make_tokendb.${ngram}gram.log 2>&1 &
done
wait

