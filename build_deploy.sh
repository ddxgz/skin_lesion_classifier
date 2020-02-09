
ProjectID="data-science-258408"
ImageName="skin_lesion_app"
Region="us-central1"

gcloud builds submit --tag gcr.io/$ProjectID/$ImageName

gcloud run deploy --platform managed --region $Region --image gcr.io/$ProjectID/$ImageName