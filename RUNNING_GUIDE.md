python pretrain.py --params-file templates/stage1_cit_con_deg.json
python pretrain.py --params-file templates/stage1_cit_cov_word.json
python pretrain.py --params-file templates/stage1_cora_con_deg.json
python pretrain.py --params-file templates/stage1_cora_cov_word.json
python pretrain.py --params-file templates/stage1_pub_con_deg.json
python pretrain.py --params-file templates/stage1_pub_cov_word.json


python run_gate_combinations.py --ssl-pool homonode,propagation,bootstrap --choose 3 --stage2json templates/stage2_cit_con_deg.json




python run_gate_combinations.py --ssl-pool hardcontrast,maskedgen,pseudolabel,bootstrap,homonode,consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_cit_con_deg.json
python run_gate_combinations.py --ssl-pool hardcontrast,maskedgen,pseudolabel,bootstrap --choose 1 --stage2json templates/stage2_cit_con_deg.json
python run_gate_combinations.py --ssl-pool hardcontrast,maskedgen,pseudolabel,bootstrap,homonode,consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_cit_cov_word.json
python run_gate_combinations.py --ssl-pool hardcontrast,maskedgen,pseudolabel,bootstrap --choose 1 --stage2json templates/stage2_cit_cov_word.json
python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_cora_con_deg.json
python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_cora_cov_word.json
python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_pub_con_deg.json
python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_pub_cov_word.json


python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 1 --stage2json templates/stage2_cit_con_deg.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_cit_con_deg.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 1 --stage2json templates/stage2_cit_cov_word.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_cit_cov_word.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 1 --stage2json templates/stage2_cora_con_deg.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_cora_con_deg.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 1 --stage2json templates/stage2_cora_cov_word.json
python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_cora_cov_word.json

python run_gate_combinations.py --ssl-pool homonode,homottt,propagation,bootstrap,pseudolabel --choose 1 --stage2json templates/stage2_pub_cov_word.json


python run_gate_combinations.py --ssl-pool entropy,hardcontrast,smoothness,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_pub_cov_word.json

python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/degree/concept  --stage3-params templates/stage3_cit_con_deg.json   --ttt-script ttt_shared.py --summary-json ./outputs/stage2/citeseer/degree/concept/my_batch_summary_shared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/word/covariate  --stage3-params templates/stage3_cit_cov_word.json  --ttt-script ttt_shared.py --summary-json ./outputs/stage2/citeseer/word/covariate/my_batch_summary_shared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/degree/concept      --stage3-params templates/stage3_cora_con_deg.json  --ttt-script ttt_shared.py --summary-json ./outputs/stage2/cora/degree/concept/my_batch_summary_shared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/word/covariate      --stage3-params templates/stage3_cora_cov_word.json --ttt-script ttt_shared.py --summary-json ./outputs/stage2/cora/word/covariate/my_batch_summary_shared.json




python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_pub_con_deg.json
python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 1 --stage2json templates/stage2_pub_cov_word.json



python run_gate_combinations.py --ssl-pool graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 3 --stage2json templates/stage2_cit_con_deg.json
python run_gate_combinations.py --ssl-pool hardcontrast,homonode,propagation,recon,homottt,maskedgen,neighbor --choose 3 --stage2json templates/stage2_cit_cov_word.json
python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 3 --stage2json templates/stage2_cora_con_deg.json
python run_gate_combinations.py --ssl-pool consistency,contrastive,degree,denoise,generative,graphtta,homottt,neighbor,propagation,recon,residual,smoothness,entropy --choose 3 --stage2json templates/stage2_cora_cov_word.json

python run_gate_combinations.py --ssl-pool hardcontrast,pseudolabel,bootstrap,homonode,consistency,contrastive,degree,homottt,propagation,recon,residual,smoothness,entropy --choose 3 --stage2json templates/stage2_pub_con_deg.json

python run_gate_combinations.py --ssl-pool hardcontrast,pseudolabel,bootstrap,homonode,degree,homottt,propagation,recon,residual,smoothness,entropy --choose 3 --stage2json templates/stage2_pub_con_deg.json



python run_gate_combinations.py --ssl-pool hardcontrast,neighbor,propagation,pseudolabel,recon,bootstrap,degree,entropy --choose 3 --stage2json templates/stage2_pub_cov_word.json


hardcontrast,homonode,homottt,propagation,neighbor, pseudolabel,degree


python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/degree/concept  --stage3-params templates/stage3_cit_con_deg.json   --summary-json ./outputs/stage2/citeseer/degree/concept/my_batch_summary.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/word/covariate  --stage3-params templates/stage3_cit_cov_word.json  --summary-json ./outputs/stage2/citeseer/word/covariate/my_batch_summary.json





python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/degree/concept  --stage3-params templates/stage3_cit_con_deg.json   --summary-json ./outputs/stage2/citeseer/degree/concept/my_batch_summary.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/word/covariate  --stage3-params templates/stage3_cit_cov_word.json  --summary-json ./outputs/stage2/citeseer/word/covariate/my_batch_summary.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/degree/concept      --stage3-params templates/stage3_cora_con_deg.json  --summary-json ./outputs/stage2/core/degree/concept/my_batch_summary.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/word/covariate      --stage3-params templates/stage3_cora_cov_word.json --summary-json ./outputs/stage2/core/word/covariate/my_batch_summary.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/pubmed/degree/concept    --stage3-params templates/stage3_pub_con_deg.json   --summary-json ./outputs/stage2/pubmed/degree/concept/my_batch_summary.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/pubmed/word/covariate    --stage3-params templates/stage3_pub_cov_word.json  --summary-json ./outputs/stage2/pubmed/word/covariate/my_batch_summary.json

python run_ttt_batch_from_stage2.py \
  --stage2-dir ./outputs/stage2/citeseer/word/covariate \
  --stage3-params templates/stage3.json \
  --summary-json ./outputs/stage2/citeseer/word/covariate/my_batch_summary.json


python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/degree/concept  --stage3-params templates/stage3_cit_con_deg.json   --ttt-script ttt_unshared.py --summary-json ./outputs/stage2/citeseer/degree/concept/my_batch_summary_unshared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/citeseer/word/covariate  --stage3-params templates/stage3_cit_cov_word.json  --ttt-script ttt_unshared.py --summary-json ./outputs/stage2/citeseer/word/covariate/my_batch_summary_unshared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/degree/concept      --stage3-params templates/stage3_cora_con_deg.json  --ttt-script ttt_unshared.py --summary-json ./outputs/stage2/core/degree/concept/my_batch_summary_unshared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/core/word/covariate      --stage3-params templates/stage3_cora_cov_word.json --ttt-script ttt_unshared.py --summary-json ./outputs/stage2/core/word/covariate/my_batch_summary_unshared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/pubmed/degree/concept    --stage3-params templates/stage3_pub_con_deg.json   --ttt-script ttt_unshared.py --summary-json ./outputs/stage2/pubmed/degree/concept/my_batch_summary_unshared.json
python run_ttt_batch_from_stage2.py --stage2-dir ./outputs/stage2/pubmed/word/covariate    --stage3-params templates/stage3_pub_cov_word.json  --ttt-script ttt_shared.py --summary-json ./outputs/stage2/pubmed/word/covariate/my_batch_summary_shared.json

python run_ttt_batch_from_stage2.py \
  --stage2-dir ./outputs/stage2/citeseer/word/covariate \
  --stage3-params templates/stage3.json \
  --max-runs 10

python run_ttt_batch_from_stage2.py \
  --stage2-dir ./outputs/stage2/citeseer/word/covariate \
  --stage3-params templates/stage3.json \
  --summary-json ./outputs/stage2/citeseer/word/covariate/my_batch_summary.json


  python run_gate_combinations.py --ssl-pool entropy,hardcontrast,smoothness,propagation,bootstrap,pseudolabel --choose 3 --stage2json templates/stage2_pub_cov_word.json