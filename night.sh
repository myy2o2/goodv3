python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 1 --stage2json templates/stage2_cora_con_deg.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_cora_con_deg.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 1 --stage2json templates/stage2_cora_cov_word.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_cora_cov_word.json

python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/degree/concept      --stage3-params templates/stage3_cora_con_deg.json  --ttt-script ttt_shared.py --summary-json ./outputs/stage2/cora/degree/concept/my_batch_summary_shared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/word/covariate      --stage3-params templates/stage3_cora_cov_word.json --ttt-script ttt_shared.py --summary-json ./outputs/stage2/cora/word/covariate/my_batch_summary_shared.json
