import pathlib
import logcorrect
import image_downloader
import os

def createFoldersIfDoesntExits(galaxy_type, solution_dir):
    if not os.path.exists("%s/images/%s/clean" % (solution_dir, galaxy_type)):
        os.makedirs("%s/images/%s/clean" % (solution_dir, galaxy_type))

def correctGalaxies(galaxy_type, solution_dir):
    galaxies = os.listdir("%s/images/%s/original" % (solution_dir, galaxy_type))
    createFoldersIfDoesntExits(galaxy_type, solution_dir)
    corrected_galaxies = 0
    for g in galaxies:
        if (g.__contains__("DR9")):
            logcorrect.removeandsave("%s/images/%s/" % (solution_dir, galaxy_type), g)
            corrected_galaxies = corrected_galaxies + 1
            if ( (corrected_galaxies*2) % 1000 == 0):
                print("Corrected galaxies: %d" % (corrected_galaxies*2))

def correctBrightness():
    project_dir = os.path.dirname(__file__)
    solution_dir = pathlib.Path(project_dir).parent
    if not os.path.exists("%s/images/spirals" % solution_dir):
        print("Downloading images...")
        image_downloader.download_images()

    correctGalaxies("spirals", solution_dir)
    correctGalaxies("ellipticals", solution_dir)
    correctGalaxies("ringed_spirals", solution_dir)
    correctGalaxies("non_ringed_spirals", solution_dir)

def main():
    correctBrightness()

if __name__ == "__main__":
    main()