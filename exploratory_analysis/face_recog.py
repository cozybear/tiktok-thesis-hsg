import cv2
from PIL import Image
import face_recognition
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import shutil
from scipy.spatial.distance import cdist
from natsort import natsort_keygen
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import DBSCAN
import hashlib
import gc


class VideoPreparation:
    def __init__(self, pathy):
        self.pathy = pathy

    def cut_videos(self):
        # for folder in metadata test
        for directory in next(os.walk(self.pathy))[1]:
            folder = ''.join(directory)

            #for video in folder videos
            for filename in os.listdir(os.path.join(self.pathy, folder, 'videos')):

                # create frames folder
                if not os.path.exists(os.path.join(self.pathy, folder, 'videos', 'frames')):
                    os.makedirs(os.path.join(self.pathy, folder, 'videos', 'frames'))

                if not os.path.exists(os.path.splitext(os.path.join(self.pathy, folder, 'videos', 'frames', filename))[0]):
                    os.makedirs(os.path.splitext(os.path.join(self.pathy, folder, 'videos', 'frames', filename))[0])
                pathOut = os.path.splitext(os.path.join(self.pathy, folder, 'videos', 'frames', filename))[0]
                #if MPEG 4 videos
                if filename.endswith(".mp4"):
                    cap = cv2.VideoCapture(os.path.join(self.pathy, folder, 'videos', filename))
                    #check wether video is opened
                    if cap.isOpened():
                        current_frame = 0
                        #define when to cut: every n frame (videos have 60 fps)
                        n = 50
                        success = True
                        while success:
                            success, image = cap.read()
                            #print('Read a new frame:', success)
                            if success:
                                if current_frame%n == 0:
                                    cv2.imwrite(os.path.join(pathOut, f'frame{current_frame}.jpg'), image)
                                current_frame += 1
                        cap.release()


            print(f'Cut videos done for {os.path.join(self.pathy, folder)}')




##########################
    def get_faces(self):
        # for folder in metadata test
        for directory in next(os.walk(self.pathy))[1]:
            folder = ''.join(directory)
            if not folder.startswith('.') and not os.path.exists(os.path.join(self.pathy, folder, 'videos', 'faces')):
                os.makedirs(os.path.join(self.pathy, folder, 'videos', 'faces'))
            # for video in folder videos

                for filename in os.listdir(os.path.join(self.pathy, folder, 'videos')):
                    if not filename.startswith('.') and not os.path.isdir(os.path.join(self.pathy, folder, 'videos', filename)):
                        filename = os.path.splitext(filename)[0]
                        if not os.path.exists(os.path.join(self.pathy, folder, 'videos', 'faces', filename)):
                            os.makedirs(os.path.join(self.pathy, folder, 'videos', 'faces', filename))
                        count = 0

                        for image in os.listdir(os.path.join(self.pathy, directory, 'videos', 'frames', filename)):
                            #image_listing = os.listdir(os.path.join(self.pathy, folder, 'videos', 'frames', filenames))
                            #for fold in os.listdir(os.path.join(self.pathy, folder, 'videos', 'frames')):
                            image = ''.join(image)
                            #if os.path.isdir(filenames):
                                #for image in direc:
                                # Load the jpg file into a numpy array
                            #if os.path.isfile(os.path.join(self.pathy, folder, 'videos', 'frames', direc, image)):
                            pathOut = os.path.join(self.pathy, directory, 'videos', 'faces', filename)

                            if not image.startswith('.'):
                                i = face_recognition.load_image_file(os.path.join(self.pathy, folder, 'videos',
                                                                                  'frames', filename, image))
                                # Find all the faces in the image using the default HOG-based model.
                                # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
                                # See also: find_faces_in_picture_cnn.py
                                face_locations = face_recognition.face_locations(i)
                                for face_location in face_locations:
                                    #print(face_location)
                                    # Print the location of each face in this image
                                    top, right, bottom, left = face_location
                                    # You can access the actual face itself like this:
                                    face_image = i[top:bottom, left:right]
                                    pil_image = Image.fromarray(face_image)
                                    # display(pil_image)
                                    pil_image.save(os.path.join(pathOut, f'face_{count}_{filename}.jpg'))
                                    count += 1

            print(f'Get faces done for {os.path.join(self.pathy, folder)}\n')

#Preparator = VideoPreparation('metadata_test')


    def filter_channel_owner(self):
        # for folder in metadata test
        for directory in next(os.walk(self.pathy))[1]:
            folder = ''.join(directory)
            total_encodings = []
            df_total = pd.DataFrame()
            total_data = []
            try:
                if not folder.startswith('.'):
                    for filename in os.listdir(os.path.join(self.pathy, folder, 'videos')):
                        if not filename.startswith('.') and not os.path.isdir(os.path.join(self.pathy, folder, 'videos', filename)):
                            try:
                                filename = os.path.splitext(filename)[0]


                                known_images = os.listdir(os.path.join(self.pathy, directory, 'videos', 'faces', filename))

                                known_encodings = []
                                data = []



                                for known_image in known_images:
                                    #exclude hidden files
                                    if not str(known_image).startswith('.'):
                                        # Load some images to compare against
                                        image = face_recognition.load_image_file(os.path.join(self.pathy, directory, 'videos',
                                                                                              'faces', filename, str(known_image)))
                                        # Get the face encodings for the known images
                                        height, width = image.shape[:2]
                                        face_encodings = face_recognition.face_encodings(image,
                                                                                         known_face_locations=[(0, width,
                                                                                                                height, 0)])

                                        data.append(os.path.basename(known_image))
                                        known_encodings.extend(face_encodings)

                                total_encodings.extend(known_encodings)
                                total_data.extend(data)
                            except:
                                #no face folder with video filename
                                continue

                # if list of encodings is not empty (faces detected)
                total_distances = []

                if total_encodings:
                    #convert list of arrays to array
                    array_encodings = np.vstack(total_encodings)

                    # calculate face encoding where sum of euclidian distances to all other encodings is minimum
                    base_face_encoding = total_encodings[cdist(array_encodings, array_encodings, 'euclidean').sum(axis=1).argmin()]

                    # for every encoded image check distance to reference
                    for encoding in total_encodings:

                        # See how far apart the test image is from the known faces
                        total_face_distance = face_recognition.face_distance([encoding], base_face_encoding)
                        total_distances.append(total_face_distance)

                    df_total['name'] = pd.Series(total_data, dtype=pd.StringDtype())
                    df_total['user'] = folder
                    df_total['distance'] = pd.Series(total_distances)
                    ########


                    # set threshold
                    threshold = 0.5

                    #
                    df_total['group'] = np.where(df_total['distance'] <= threshold, 'Channel Owner', 'Supporting Actors')

                    df_total['face_encoding'] = total_encodings

                    path = os.path.join(self.pathy, folder, 'videos', 'inandout_total')

                    # sort df by column 'name' and save to file
                    df_total.sort_values(by='name', key=natsort_keygen()).reset_index(drop=True).to_csv(path)

                    df_cleaned = df_total[df_total['group'] == 'Channel Owner'].copy()
                    df_cleaned['path_to_image'] = df_cleaned.apply(lambda row: os.path.join(self.pathy, folder, 'videos', 'faces_cleaned', row['name']), axis=1)
                    path = os.path.join(self.pathy, folder, 'videos', 'inandout_cleaned')
                    df_cleaned.sort_values(by='name', key=natsort_keygen()).reset_index(drop=True).to_csv(path)
                    ###############



                    # make a TSNE scatterplot of encodings
                    X = np.array(total_encodings)

                    model = TSNE(n_components=2, init='random', perplexity=17)

                    tsne_data = model.fit_transform(X)

                    df_tsne = pd.DataFrame()
                    df_tsne["Affiliation"] = df_total['group']
                    df_tsne["t-SNE_1"] = tsne_data[:, 0]
                    df_tsne["t-SNE_2"] = tsne_data[:, 1]


                    sns.set(rc={'figure.figsize': (11.7, 8.27)})
                    sns.scatterplot(x="t-SNE_1", y="t-SNE_2", hue="Affiliation",
                                    data=df_tsne, legend='full', palette=['orangered', 'lime']).set(
                        title="Face Vector Embeddings t-SNE Projection")
                    plt.style.use('ggplot')
                    path = os.path.join(self.pathy, folder, 'videos',
                                        'tsneplot_total')
                    plt.savefig(path, dpi=400)
                    # clear figure
                    plt.clf()

                    os.makedirs(os.path.join(self.pathy, folder, 'videos', 'faces_cleaned'))
                    with open(os.path.join(self.pathy, folder, 'videos', 'inandout_total')) as inandout:
                        table = pd.read_csv(inandout)
                        # exclude 'Supporting Actors' rows
                        table = table[~table.group.str.contains("Supporting Actors")]
                        # for row in table, copy face to folder
                        for filename in os.listdir(os.path.join(self.pathy, folder, 'videos')):
                            if not filename.startswith('.') and not os.path.isdir(
                                    os.path.join(self.pathy, folder, 'videos', filename)):
                                filename = os.path.splitext(filename)[0]
                                for index, row in table.iterrows():
                                    try:
                                        shutil.copyfile(os.path.join(self.pathy, folder, 'videos', 'faces', filename, row['name']),
                                                        os.path.join(self.pathy, folder, 'videos', 'faces_cleaned',
                                                                     row['name']))

                                    except:
                                        continue

                print(f'Get channel owner and tsne done for {os.path.join(self.pathy, folder)}')
            except Exception as e:
                print(e)
                print('no faces')
                pass

    def k_means_filter(self):
        # for folder in metadata test
        for directory in next(os.walk(self.pathy))[1]:
            folder = ''.join(directory)
            total_encodings = []
            df_total = pd.DataFrame()
            total_data = []
            try:
                if not folder.startswith('.'):
                    for filename in os.listdir(os.path.join(self.pathy, folder, 'videos')):
                        if not filename.startswith('.') and not os.path.isdir(
                                os.path.join(self.pathy, folder, 'videos', filename)):
                            try:
                                filename = os.path.splitext(filename)[0]

                                known_images = os.listdir(os.path.join(self.pathy, directory, 'videos', 'faces', filename))

                                known_encodings = []
                                data = []

                                for known_image in known_images:
                                    # exclude hidden files
                                    if not str(known_image).startswith('.'):
                                        # Load some images to compare against
                                        image = face_recognition.load_image_file(
                                            os.path.join(self.pathy, directory, 'videos',
                                                         'faces', filename, str(known_image)))
                                        # Get the face encodings for the known images
                                        height, width = image.shape[:2]
                                        face_encodings = face_recognition.face_encodings(image,
                                                                                         known_face_locations=[(0, width,
                                                                                                                height, 0)])

                                        data.append(os.path.basename(known_image))
                                        known_encodings.extend(face_encodings)

                                total_encodings.extend(known_encodings)
                                total_data.extend(data)
                            except:
                                #no face folder with video filename
                                continue
                if total_encodings:

                    distortions = []
                    K = range(1, 10)
                    for k in K:
                        kmeanModel = KMeans(n_clusters=k)
                        kmeanModel.fit(total_encodings)
                        distortions.append(kmeanModel.inertia_)

                    #plot elbow method


                    x = range(1, len(distortions) + 1)

                    kn = KneeLocator(x, distortions, curve='convex', direction='decreasing')
                    plt.xlabel('Number of Clusters k')
                    plt.ylabel('Sum of Squared Distances')
                    plt.title('Elbow Method Analysis for Optimal k')
                    plt.plot(x, distortions, 'bx-')
                    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

                    path = os.path.join(self.pathy, folder, 'videos',
                                        'k_means_elbow')
                    plt.savefig(path, dpi=400)
                    # clear figure
                    plt.clf()

                    #choose optimal k
                    best_k = kn.knee

                    #random init
                    n_random = random.randint(0, 9999)

                    #init kMeans
                    kmeanModel = KMeans(n_clusters=best_k, random_state=n_random)
                    kmeanModel.fit(total_encodings)

                    labels = kmeanModel.labels_

                    df_total['name'] = pd.Series(total_data, dtype=pd.StringDtype())
                    df_total['group'] = labels
                    df_total['encodings'] = total_encodings

                    #majority cluster is channel owner
                    majority = df_total['group'].mode()[0]
                    df_total['group'] = df_total['group'].replace(majority, 'Channel Owner')

                    df_total['group'] = np.where(df_total['group'] != 'Channel Owner', 'Supporting Actor ' + df_total['group'].astype('category').cat.codes.add(1).astype(str), df_total['group'])
                    path = os.path.join(self.pathy, folder, 'videos', 'kmeans_inandout_total')

                    # sort df by column 'name' and save to file
                    df_total.sort_values(by='name', key=natsort_keygen()).reset_index(drop=True).to_csv(path)



                    # make a TSNE scatterplot of encodings
                    X = np.array(total_encodings)

                    # TODO: set perplexity
                    model = TSNE(n_components=2, init='random', perplexity=30)

                    tsne_data = model.fit_transform(X)

                    df_tsne = pd.DataFrame()
                    df_tsne["Affiliation"] = df_total['group']
                    df_tsne["t-SNE_1"] = tsne_data[:, 0]
                    df_tsne["t-SNE_2"] = tsne_data[:, 1]


                    sns.set(rc={'figure.figsize': (11.7, 8.27)})
                    sns.scatterplot(x="t-SNE_1", y="t-SNE_2", hue="Affiliation",
                                    data=df_tsne, legend='full', palette=['orangered', 'lime',
                                                                          'royalblue', 'fuchsia', 'orange', 'black', 'sienna',
                                                                          'orchid']).set(
                        title="t-SNE Face Projection of k-Means Clusters")

                    plt.style.use('ggplot')
                    path = os.path.join(self.pathy, folder, 'videos',
                                        'k_means_tsneplot_total')
                    plt.savefig(path, dpi=400)
                    # clear figure
                    plt.clf()

                    os.makedirs(os.path.join(self.pathy, folder, 'videos', 'faces_cleaned'))
                    with open(os.path.join(self.pathy, folder, 'videos', 'kmeans_inandout_total')) as inandout:
                        table = pd.read_csv(inandout)
                        # exclude 'Supporting Actors' rows
                        table = table[~table.group.str.contains("Supporting Actor")]
                        # for row in table, copy face to folder
                        for filename in os.listdir(os.path.join(self.pathy, folder, 'videos')):
                            if not filename.startswith('.') and not os.path.isdir(
                                    os.path.join(self.pathy, folder, 'videos', filename)):
                                filename = os.path.splitext(filename)[0]
                                for index, row in table.iterrows():
                                    try:
                                        shutil.copyfile(
                                            os.path.join(self.pathy, folder, 'videos', 'faces', filename, row['name']),
                                            os.path.join(self.pathy, folder, 'videos', 'faces_cleaned',
                                                         row['name']))

                                    except:
                                        continue
                
                    print(f'Get channel owner and k-means done for {os.path.join(self.pathy, folder)}')
            except Exception as e:
                print(e)
                print('no faces')
                pass

    def dhash(self, i, hash_size=8):
        # Grayscale and shrink the image in one step.
        i = i.convert('L').resize(
            (hash_size + 1, hash_size),
            Image.ANTIALIAS,
        )
        pixels = list(i.getdata())  # Compare adjacent pixels.
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = i.getpixel((col, row))
                pixel_right = i.getpixel((col + 1, row))
                difference.append(pixel_left > pixel_right)  # Convert the binary array to a hexadecimal string.
        decimal_value = 0
        hex_string = []
        for index, value in enumerate(difference):
            if value:
                decimal_value += 2 ** (index % 8)
            if (index % 8) == 7:
                hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
                decimal_value = 0
        return ''.join(hex_string)

    def DBSCAN_filter(self):

        # for folder in metadata test
        for directory in next(os.walk(self.pathy))[1]:
            folder = ''.join(directory)
            total_encodings = []
            df_total = pd.DataFrame()
            total_data = []
            images = []

            try:
                if not folder.startswith('.'):
                    for filename in os.listdir(os.path.join(self.pathy, folder, 'videos')):
                        if not filename.startswith('.') and not filename.startswith('dbscan') and not os.path.isdir(
                                os.path.join(self.pathy, folder, 'videos', filename)):
                            try:
                                #get image encodings for all images in all directories
                                filename = os.path.splitext(filename)[0]

                                known_images = os.listdir(
                                    os.path.join(self.pathy, directory, 'videos', 'faces', filename))


                                known_encodings = []
                                data = []
                                hashes = []

                                for known_image in known_images:
                                    # exclude hidden files
                                    if not str(known_image).startswith('.'):
                                        #hash images and only append if not already used
                                        image_file = Image.open(os.path.join(self.pathy, directory, 'videos',
                                                         'faces', filename, str(known_image)))



                                        #hash image
                                        hash = self.dhash(image_file)

                                        image_file.close()

                                        del image_file
                                        gc.collect()

                                        #when there is not already an identical picture
                                        if not hash in hashes:
                                            hashes.append(hash)
                                            # get pathname of image and add to list
                                            images.append(os.path.join(self.pathy, folder, 'videos', 'faces', filename, str(known_image)))

                                            # Load some images to compare against
                                            image = face_recognition.load_image_file(
                                                os.path.join(self.pathy, directory, 'videos',
                                                             'faces', filename, str(known_image)))
                                            # Get the face encodings for the known images
                                            height, width = image.shape[:2]
                                            face_encoding = face_recognition.face_encodings(image,
                                                                                             known_face_locations=[
                                                                                                 (0, width,
                                                                                                  height, 0)])
                                            known_encodings.extend(face_encoding)
                                            data.append(os.path.basename(known_image))

                                total_encodings.extend(known_encodings)
                                total_data.extend(data)

                            except Exception as e:
                                print(e)
                                # no face folder with video filename
                                continue

                if total_encodings:

                    clustering = DBSCAN(eps=.5, metric='euclidean', min_samples=1).fit(total_encodings)
                    labels = clustering.labels_


                    df_total['name'] = pd.Series(total_data, dtype=pd.StringDtype())
                    df_total['group'] = labels
                    df_total['encodings'] = total_encodings

                    # majority cluster is channel owner

                    majority = df_total['group'].mode()[0]
                    df_total['group'] = df_total['group'].replace(majority, 'Channel Owner')

                    df_total['group'] = np.where(df_total['group'] != 'Channel Owner',
                                                 'Supporting Actor ' + df_total['group'].astype(
                                                     'category').cat.codes.add(1).astype(str), df_total['group'])


                    path = os.path.join(self.pathy, folder, 'videos', 'dbscan_inandout_total')
                    # sort df by column 'name' and save to file
                    df_total.sort_values(by='name', key=natsort_keygen()).reset_index(drop=True).to_csv(path)


                    """
                    # make a TSNE scatterplot of encodings
                    X = np.array(total_encodings)

                    # TODO: set perplexity
                    model = TSNE(n_components=2, init='random', perplexity=14)

                    tsne_data = model.fit_transform(X)

                    df_tsne = pd.DataFrame()
                    df_tsne["Affiliation"] = df_total['group']
                    df_tsne["t-SNE_1"] = tsne_data[:, 0]
                    df_tsne["t-SNE_2"] = tsne_data[:, 1]



                    fig = plt.figure(figsize=(15, 10), num=1, clear=True)
                    plt.style.use('ggplot')
                    ax = fig.add_subplot(aspect='equal')

                    palette = sns.color_palette("bright", np.max(labels) + 1)

                    sns.scatterplot(x="t-SNE_1", y="t-SNE_2", hue="Affiliation",
                                    data=df_tsne, legend='full', ax=ax, palette=palette).set(
                        title="t-SNE Projections of DBSCAN Clusters")

                    _ = ax.axis('tight')

                    path = os.path.join(self.pathy, folder, 'videos',
                                        'dbscan_tsneplot_total')
                    fig.savefig(path, dpi=400)




                    colors = [palette[tsne_data] if tsne_data >= 0 else (0, 0, 0) for tsne_data in labels]

                    for i in range(len(images)):
                        image_ann = plt.imread(images[i])
                        im = OffsetImage(image_ann, zoom=.15)
                        bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
                        ab = AnnotationBbox(im, tsne_data[i], xycoords='data',
                                            frameon=(bboxprops is not None),
                                            pad=0.02,
                                            bboxprops=bboxprops)
                        ax.add_artist(ab)

                    path = os.path.join(self.pathy, folder, 'videos',
                                        'dbscan_tsneplot_total_images')
                    fig.savefig(path, dpi=400)
                    # clear figure
                    plt.clf()
                    #close figure
                    plt.close('all')
                    """

                    #create "cleaned" directory
                    if not os.path.exists(os.path.join(self.pathy, folder, 'videos', 'faces_cleaned')):
                        os.makedirs(os.path.join(self.pathy, folder, 'videos', 'faces_cleaned'))

                    #delete all contents from directory
                    for file in os.listdir(os.path.join(self.pathy, folder, 'videos', 'faces_cleaned')):
                        os.unlink(os.path.join(self.pathy, folder, 'videos', 'faces_cleaned', file))

                    #save cleaned faces to directory
                    with open(os.path.join(self.pathy, folder, 'videos', 'dbscan_inandout_total')) as inandout:
                        table = pd.read_csv(inandout)
                        # exclude 'Supporting Actors' rows
                        table = table[~table.group.str.contains("Supporting Actor")]
                        # for row in table, copy face to folder
                        for f in os.listdir(os.path.join(self.pathy, folder, 'videos')):
                            if not f.startswith('.') and not os.path.isdir(
                                    os.path.join(self.pathy, folder, 'videos', f)):
                                f = os.path.splitext(f)[0]
                                for index, row in table.iterrows():
                                    try:
                                        shutil.copyfile(
                                            os.path.join(self.pathy, folder, 'videos', 'faces', f, row['name']),
                                            os.path.join(self.pathy, folder, 'videos', 'faces_cleaned',
                                                         row['name']))

                                    except:
                                        continue

                print(f'Get channel owner and DBSCAN done for {os.path.join(self.pathy, folder)}')



            except Exception as e:
                print(e)
                continue


filter = VideoPreparation('/Volumes/My_Passport/tiktokdata/metadata_videos')
filter.cut_videos()
filter.get_faces()

#filter.filter_channel_owner()
#filter.k_means_filter()


filter.DBSCAN_filter()



