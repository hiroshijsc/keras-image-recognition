import urllib.request

url_files = ["", ""]
for url in url_files:
    savename = url.split("/")[-1]
    urllib.request.urlretrieve(url, savename)
    print("Complete")
