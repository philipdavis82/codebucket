#start python code
import git
from os import path
repo1 = "git://github.com/reggi/fortune-cookie.git"
repo2 = "git://github.com/ruanyf/fortunes.git"
repo1_dir = "fortune-cookie/"
repo2_dir = "fortune/s"
main_dir = "cookie_data/"

if not path.exists(main_dir+repo1_dir):
    git.Git("C:\Software\Fortune Cookie Writer\cookie_data").clone(repo1)
if not path.exists(main_dir+repo2_dir):
    git.Git("C:\Software\Fortune Cookie Writer\cookie_data").clone(repo2)

data_paths = {
    "repo1" : main_dir+repo1_dir+"fortune-cookies.txt",
    "repo2" : main_dir+repo2_dir+"fortunes"
}