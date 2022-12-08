import pathlib
import image_generator
import background_maker
import os

def createFoldersIfDoesntExits(galaxy_type, solution_dir):
    if not os.path.exists("%s/images/%s/corrected" % (solution_dir, galaxy_type)):
        os.makedirs("%s/images/%s/corrected" % (solution_dir, galaxy_type))
    if not os.path.exists("%s/images/%s/clean" % (solution_dir, galaxy_type)):
        os.makedirs("%s/images/%s/clean" % (solution_dir, galaxy_type))
    if not os.path.exists("%s/images/%s/cropped" % (solution_dir, galaxy_type)):
        os.makedirs("%s/images/%s/cropped" % (solution_dir, galaxy_type))

def generate(solution_dir):
    if os.path.exists("%s/images" % solution_dir):
        for i in range(2560):
            image_generator.generateSkyImage(solution_dir, i, 0)
        for i in range(2560, 2816):
            image_generator.generateSkyImage(solution_dir, i, 1)
        for i in range(2816, 3072):
            image_generator.generateSkyImage(solution_dir, i, 2)        

def main():
    project_dir = os.path.dirname(__file__)
    solution_dir = pathlib.Path(project_dir).parent

    createFoldersIfDoesntExits("spirals", solution_dir)
    createFoldersIfDoesntExits("ellipticals", solution_dir)
    createFoldersIfDoesntExits("non_ringed_spirals", solution_dir)
    createFoldersIfDoesntExits("ringed_spirals", solution_dir)

    generate(solution_dir)
    # background_maker.generate_backgrounds()

if __name__ == "__main__":
    main()