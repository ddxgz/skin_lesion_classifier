import subprocess

import click
from webapp.app import app

ProjectID = "data-science-258408"
ImageName = "skin_lesion_app"
Region = "us-central1"
ImageBuild = f"gcr.io/{ProjectID}/{ImageName}"
ServiceName = "skinlesionapp"


@click.group()
def cli():
    pass


@cli.command()
@click.option('--local/--container', default=True)
def run(local: bool):
    if local:
        app.debug = True
        # app.run(host='0.0.0.0', threaded=True)
        app.run(host='0.0.0.0:5000')
    else:
        run_container()

#         docker build --tag $ImageBuild . && docker run -it -p 5000:80 $ImageBuild
# gcloud builds submit --tag gcr.io/$ProjectID/$ImageName
@cli.command()
@click.option('--local/--cloud', default=True)
@click.option('--run', is_flag=True, default=False)
@click.option('--deploy', is_flag=True, default=False)
def build(local: bool, run: bool, deploy: bool):
    if local:
        build_local()
    else:
        build_gcloud()

    if run:
        run_container()

    if deploy:
        deploy_gcloud()


#        docker push $ImageBuild
#        gcloud run deploy $ServiceName --platform managed --region $Region --image $ImageBuild
@cli.command()
def deploy():
    deploy_gcloud()


def run_container():
    print('Running image')
    subprocess.run(['docker', 'run', '-it', '-p', '5000:80', ImageBuild])


def build_local():
    print('Building image locally')
    subprocess.run(['docker', 'build', '--tag', ImageBuild, '.'])


def build_gcloud():
    print('Building image in Google Cloud Build')
    subprocess.run(['gcloud', 'builds', 'submit',
                    '--tag', ImageBuild])


def deploy_gcloud():
    print('Deploying image')
    subprocess.run(['docker', 'push', ImageBuild])
    subprocess.run(['gcloud', 'run', 'deploy',
                    ServiceName, '--platform', 'managed', '--region', Region, '--image', ImageBuild])

# ModelPath="models/exp-dense161_eq3_exclutest_lesion_mcc_v1_model_best.pth.tar"
# Bucket="data-science-258408-skin-lesion-cls-models"
# ModelCloud="models/dense161.pth.tar"
#
# gsutil cp $ModelPath gs://$Bucket/$ModelCloud
# # https://storage.cloud.google.com/data-science-258408-skin-lesion-cls-models/models/dense161.pth.tar
# @cli.command()
# def upload():
#     pass


if __name__ == '__main__':
    cli()
