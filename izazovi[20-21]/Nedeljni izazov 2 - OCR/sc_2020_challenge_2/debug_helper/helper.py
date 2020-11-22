import matplotlib.pylab as plt

def plot_images(images):
    """
    Help to debug:https://stackoverflow.com/questions/19471814/display-multiple-images-in-one-ipython-notebook-cell
    """
    plt.figure(figsize=(30,20))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images)/columns + 1, columns, i + 1)
        plt.imshow(image)


# kreiranje recnika svih poznatih reci, za korekciju Levenstein rastojanjem
VOCABULARY_PATH = 'dataset/dict.txt'
vocabulary = dict()
with open(VOCABULARY_PATH, 'r', encoding='utf-8') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        cols = line.split()
        if len(cols) == 3:
            vocabulary[cols[1]] = cols[2]