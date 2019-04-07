from os import path

class parser():
    class lnln():
        def __init__(self,filename):
            self.filename = filename
            self.fortunes = []

        def load_list(self):
            File = open(self.filename)
            for line in File:
                #print(line)
                line = line.replace("\n","")
                self.fortunes.append(line)
            #print(self.fortunes)
            return self.fortunes
            pass

    class prlnds():
        def __init__(self,filename):
            self.filename = filename
            self.fortunes = []

        def load_list(self):
            File = open(self.filename,'r',)
            for line in File:
                
                line = line.replace("\n","")
                if  "%" not in line and \
                    "--" not in line and \
                    ":" not in line and \
                    "[" not in line and \
                    "pp." not in line and\
                    "as quoted" not in line and\
                    line:
                    
                    self.fortunes.append(line)
            #for fortune in self.fortunes:
            #    print(fortune)
            return self.fortunes
            pass
    
    def __init__(self):
        self.paths = [
            ("lnln","cookie_data/fortune-cookie/fortune-cookies.txt"),
            ("prlnds","cookie_data/fortunes/data/fortunes")
        ]
        self.parsers = {
            "lnln":self.lnln,
            "prlnds":self.prlnds
        }
        self.main_path = "cookie_data/full_set"
    
    def load_from_main_data(self):
        if path.exists(self.main_path):
            File = open(self.main_path,'r')
            Fortunes = []
            for line in File:
                if line:
                    line = line.replace("\n","")
                    Fortunes.append(line)
            return Fortunes
        else:
            raise Exception("No Path")

    def load_from_repo_data(self):
        self.repos = []
        for path in self.paths:
            self.repos.append(self.parsers[path[0]](path[1]))
        self.fortunes = []
        for repo in self.repos:
            self.fortunes.extend(repo.load_list())
        print(self.fortunes[0])
        print(self.fortunes[-1])
        print("length ::",len(self.fortunes))

    def save(self,filename="cookie_data/full_set"):
        File = open(filename,'w')
        for fortune in self.fortunes:
            File.write(fortune+'\n')
        File.close()
