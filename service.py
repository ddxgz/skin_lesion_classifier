from webapp.app import app

if __name__ == '__main__':

    # fitbitapp.debug = True
    # # app.run(host='0.0.0.0', threaded=True)
    # fitbitapp.run(host='0.0.0.0')

    # ihealthapp.debug = True
    # # app.run(host='0.0.0.0', threaded=True)
    # ihealthapp.run(host='0.0.0.0')

    app.debug = True
    # app.run(host='0.0.0.0', threaded=True)
    app.run(host='0.0.0.0')
