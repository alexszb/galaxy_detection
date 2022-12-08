import pathlib
from random import random, shuffle
import requests
from requests.adapters import Retry, HTTPAdapter
import os
import gzip
import shutil
import time
from PIL import Image
from multiprocessing.pool import ThreadPool

def geturls(data_release, spirals, ellipticals, r_spirals, n_r_spirals):
    urls = []
    for k in spirals:
        url = []
        url.append(data_release)
        url.append("http://skyservice.pha.jhu.edu/"+data_release+"/ImgCutout/getjpeg.aspx?ra=" + str(k[1]) + "&dec=" + str(k[2]) + "&scale=0.396127%20%20%20&width=424&height=424")
        url.append(k[0])
        url.append('spirals/')
        urls.append(url)

    for e in ellipticals:
        url = []
        url.append(data_release)
        url.append("http://skyservice.pha.jhu.edu/"+data_release+"/ImgCutout/getjpeg.aspx?ra=" + str(e[1]) + "&dec=" + str(e[2]) + "&scale=0.396127%20%20%20&width=424&height=424")
        url.append(e[0])
        url.append('ellipticals/')
        urls.append(url)

    for p in r_spirals:
        url = []
        url.append(data_release)
        url.append("http://skyservice.pha.jhu.edu/"+data_release+"/ImgCutout/getjpeg.aspx?ra=" + str(p[1]) + "&dec=" + str(p[2]) + "&scale=0.396127%20%20%20&width=424&height=424")
        url.append(p[0])
        url.append('ringed_spirals/')
        urls.append(url)

    for q in n_r_spirals:
        url = []
        url.append(data_release)
        url.append("http://skyservice.pha.jhu.edu/"+data_release+"/ImgCutout/getjpeg.aspx?ra=" + str(q[1]) + "&dec=" + str(q[2]) + "&scale=0.396127%20%20%20&width=424&height=424")
        url.append(q[0])
        url.append('non_ringed_spirals/')
        urls.append(url)
    return urls

def download_images():

    project_dir = os.path.dirname(__file__)
    solution_dir = pathlib.Path(project_dir).parent

    if not os.path.exists("%s/images/spirals" % solution_dir):
            os.makedirs("%s/images/spirals" % solution_dir)
            os.makedirs("%s/images/spirals/original" % solution_dir)
    if not os.path.exists("%s/images/ellipticals" % solution_dir):
            os.makedirs("%s/images/ellipticals" % solution_dir)        
            os.makedirs("%s/images/ellipticals/original" % solution_dir)        
    if not os.path.exists("%s/images/ringed_spirals" % solution_dir):
            os.makedirs("%s/images/ringed_spirals" % solution_dir)
            os.makedirs("%s/images/ringed_spirals/original" % solution_dir)       
    if not os.path.exists("%s/images/non_ringed_spirals" % solution_dir):
            os.makedirs("%s/images/non_ringed_spirals" % solution_dir)
            os.makedirs("%s/images/non_ringed_spirals/original" % solution_dir)    

    spirals = []
    ellipticals = []
    ringed_spirals = []
    non_ringed_spirals = []



    if not os.path.isfile("%s/gz2_hart16.csv.gz" % project_dir):
        zippedCatalogue = requests.get("https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz")
        open("%s/gz2_hart16.csv.gz" % project_dir, "wb").write(zippedCatalogue.content)

    if not os.path.isfile("%s/gz2_hart16.csv" % project_dir):
        with gzip.open("%s/gz2_hart16.csv.gz" % project_dir, 'rb') as f_in:
            with open('%s/gz2_hart16.csv' % project_dir, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    with open ('%s/gz2_hart16.csv' % project_dir) as fp:
        lines = fp.readlines()
        for line in lines:
            galaxydata = line.split(',')
            if(galaxydata[20] == '1' and galaxydata[14] == '0' and galaxydata[26] == '0' and int(galaxydata[15]) > 10):
                spirals.append((galaxydata[0],galaxydata[1],galaxydata[2],galaxydata[15]))
                
            if(galaxydata[20] == '0' and galaxydata[14] == '1' and galaxydata[26] == '0' and int(galaxydata[9]) > 10):
                ellipticals.append((galaxydata[0],galaxydata[1],galaxydata[2],galaxydata[9]))

            
            if(galaxydata[122] == '1' and galaxydata[26] == '0' and galaxydata[20] == '1' and galaxydata[14] == '0' and int(galaxydata[117]) > 10):
                ringed_spirals.append((galaxydata[0],galaxydata[1],galaxydata[2],galaxydata[117]))


            if(galaxydata[122] == '0' and galaxydata[26] == '0' and galaxydata[20] == '1' and galaxydata[14] == '0' and int(galaxydata[15]) > 10):         
                non_ringed_spirals.append((galaxydata[0],galaxydata[1],galaxydata[2],galaxydata[15]))

    os.remove("%s/gz2_hart16.csv" % project_dir)

    # print(len(spirals), len(ellipticals), len(ringed_spirals), len(non_ringed_spirals))

    spirals = sorted(spirals, key=lambda x:x[3], reverse=True)
    ellipticals = sorted(ellipticals, key=lambda x:x[3], reverse=True)
    ringed_spirals = sorted(ringed_spirals, key=lambda x:x[3], reverse=True)
    non_ringed_spirals = sorted(non_ringed_spirals, key=lambda x:x[3], reverse=True)

    top_main = 20000
    top_sub = 5000

    train_spirals = spirals[0:top_main]
    train_ellipticals = ellipticals[0:top_main]
    train_ringed_spirals = ringed_spirals[0:top_sub]
    train_non_ringed_spirals = non_ringed_spirals[0:top_sub]

    # print(spirals[-1][3],ellipticals[-1][3],ringed_spirals[-1][3],non_ringed_spirals[-1][3])

    shuffle(train_spirals)
    shuffle(train_ellipticals)
    shuffle(train_ringed_spirals)
    shuffle(train_non_ringed_spirals)

    dr7_urls = geturls("DR7", train_spirals, train_ellipticals, train_ringed_spirals, train_non_ringed_spirals)
    dr9_urls = geturls("DR9", train_spirals, train_ellipticals, train_ringed_spirals, train_non_ringed_spirals)

    def download_image(url):
            s = requests.Session()
            retries = Retry(total=10, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
            s.mount('http://', HTTPAdapter(max_retries=retries))

            im = Image.open(s.get(url[1], stream=True).raw)
            # im = Image.open(requests.get(url[1], stream=True).raw)
            # imc = im.crop((53,53,371,371))
            saveloc = os.path.join("%s/images/" % solution_dir + url[3]+"/original", url[2]+"_"+url[0]+".jpeg")
            im.save(saveloc)    
            return url[1]

    img_downloaded = 0
    start_time = time.time()
    results = ThreadPool(10).imap_unordered(download_image, dr7_urls)
    for objid in results:
        # print(objid)
        img_downloaded = img_downloaded + 1
        if(img_downloaded % 100 == 0):
            print("Images downloaded: %d. Time passed: %d s." % (img_downloaded, int((time.time()-start_time))))
        
    results = ThreadPool(10).imap_unordered(download_image, dr9_urls)
    for objid in results:
        # print(objid)
        img_downloaded = img_downloaded + 1
        if(img_downloaded % 100 == 0):
            print("Images downloaded: %d. Time passed: %d s." % (img_downloaded, int((time.time()-start_time))))

    test_spirals = spirals[top_main:100000]
    test_ellipticals = ellipticals[top_main:100000]
    test_ringed_spirals = ringed_spirals[top_sub:100000]
    test_non_ringed_spirals = non_ringed_spirals[top_sub:100000]

    dr7_urls = geturls("DR7", test_spirals, test_ellipticals, test_ringed_spirals, test_non_ringed_spirals)
    dr9_urls = geturls("DR9", test_spirals, test_ellipticals, test_ringed_spirals, test_non_ringed_spirals)

    img_downloaded = 0
    start_time = time.time()
    results = ThreadPool(10).imap_unordered(download_image, dr7_urls)
    for objid in results:
        # print(objid)
        img_downloaded = img_downloaded + 1
        if(img_downloaded % 100 == 0):
            print("Images downloaded: %d. Time passed: %d s." % (img_downloaded, int((time.time()-start_time))))
        
    results = ThreadPool(10).imap_unordered(download_image, dr9_urls)
    for objid in results:
        # print(objid)
        img_downloaded = img_downloaded + 1
        if(img_downloaded % 100 == 0):
            print("Images downloaded: %d. Time passed: %d s." % (img_downloaded, int((time.time()-start_time))))