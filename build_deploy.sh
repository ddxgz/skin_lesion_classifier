
ProjectID="data-science-258408"
ImageName="skin_lesion_app"
Region="us-central1"
ImageBuild="gcr.io/$ProjectID/$ImageName"
ServiceName="skinlesionapp"

if [ $# -eq 1 ]
then
    DevStatus="$1"
    if [ $DevStatus == "dev" ]
    then
        # docker build -t $ImageName/dev . && docker run -it -p 5000:80 $ImageName/dev
        docker build --tag $ImageBuild . && docker run -it -p 5000:80 $ImageBuild
    elif [ $DevStatus == "build" ]
    then
        docker build --tag $ImageBuild .
    elif [ $DevStatus == "deploy" ]
    then
        # gcloud builds submit --tag gcr.io/$ProjectID/$ImageName
        # docker build . --tag $ImageBuild
        docker push $ImageBuild
        gcloud run deploy $ServiceName --platform managed --region $Region --image $ImageBuild 
    else
        echo "Wrong arg!"
    fi
else
    echo "Need 1 arg! dev, build or deploy"
fi