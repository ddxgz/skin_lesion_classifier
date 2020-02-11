# upload the model to Google cloud storage


ModelPath="models/exp-dense161_eq3_exclutest_lesion_mcc_v1_model_best.pth.tar"
Bucket="data-science-258408-skin-lesion-cls-models"
ModelCloud="models/dense161.pth.tar"

gsutil cp $ModelPath gs://$Bucket/$ModelCloud
# https://storage.cloud.google.com/data-science-258408-skin-lesion-cls-models/models/dense161.pth.tar