from bf import *

tiles=np.load("dati/tiles.npy")
masks=np.load("dati/masks_originali.npy")

tiles.shape, masks.shape, type(tiles), type(masks)

cls_dict=cls_dict_from_mask_MC(masks[10])
cls_dict

plot_img_label_MC(tiles[10],masks[10],cls_dict)

X_val, Y_val, C_val, X_trn, Y_trn, C_trn=split_train_val_MC(tiles,masks,cls_dict=cls_dict )

model = create_model_auto_grid(tiles, masks, basedir = 'Models', name="primo_modello_MC" )

model=train_model(model, X_trn, Y_trn,C_trn,X_val,Y_val,C_val,epochs=200)

Y_val_pred, res_val_pred = tuple(zip(*[model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False) for x in X_val[:]]))

i = 10
plot_img_label_MC(X_val[i],Y_val[i], C_val[i], lbl_title="label GT")
plot_img_label_MC(X_val[i],Y_val_pred[i], class_from_res(res_val_pred[i]), lbl_title="label Pred")

plot_metrics(Y_val, Y_val_pred)

i=np.random.randint(len(X_val))
print(i)
plot_img_label_MC(X_val[i],Y_val[i], C_val[i], lbl_title="label GT")
plot_img_label_MC(X_val[i],Y_val_pred[i], class_from_res(res_val_pred[i]), lbl_title="label Pred")

#masks1, masks2 =split_and_enumerate(masks)
masks1=np.load("dati/masks_SC_linfociti.npy")
masks2=np.load("dati/masks_SC_melanofagi.npy")

plt.figure(figsize=(8,8))
values = np.unique(masks[10].ravel())
im = plt.imshow(masks[10])
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="Pixel value = {l}".format(l=values[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.show()

plt.figure(figsize=(8,8))
values = np.unique(masks1[10].ravel())
im = plt.imshow(masks1[10])
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="Pixel value = {l}".format(l=values[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.show()

plt.figure(figsize=(8,8))
values = np.unique(masks2[10].ravel())
im = plt.imshow(masks2[10])
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="Pixel value = {l}".format(l=values[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.show()

rng = np.random.RandomState(42)
ind = rng.permutation(len(tiles))
n_val = max(1, int(round(0.1 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [tiles[i] for i in ind_val]  , [masks1[i] for i in ind_val]
X_trn, Y_trn = [tiles[i] for i in ind_train], [masks1[i] for i in ind_train]
print('number of images: %3d' % len(tiles))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

model = create_model_auto_grid(tiles, masks1, basedir = 'Models', name="modello_singlola_classe_1", n_classes=1)

model.train(X_trn,Y_trn,
            validation_data=(X_val,Y_val),
            augmenter=augmenter,
            epochs=200)

model.optimize_thresholds(X_val, Y_val)

i = 0
label, res = model.predict_instances(X_val[i], n_tiles=model._guess_n_tiles(X_val[i]))
Y_val_pred, res_val_pred = tuple(zip(*[model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False) for x in X_val[:]]))

plot_metrics(Y_val, Y_val_pred)

i=np.random.randint(len(X_val))
print(i)
plot_img_label_SC(X_val[i],Y_val[i], 1, lbl_title="label GT")
plt.show()
plot_img_label_SC(X_val[i],Y_val_pred[i], 1, lbl_title="label Pred")

rng = np.random.RandomState(42)
ind = rng.permutation(len(tiles))
n_val = max(1, int(round(0.1 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [tiles[i] for i in ind_val]  , [masks2[i] for i in ind_val]
X_trn, Y_trn = [tiles[i] for i in ind_train], [masks2[i] for i in ind_train]
print('number of images: %3d' % len(tiles))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))
model = create_model_auto_grid(tiles, masks1, basedir = 'Models', name="modello_singlola_classe_2", n_classes=1)
model.train(X_trn,Y_trn,
            validation_data=(X_val,Y_val),
            augmenter=augmenter,
            epochs=200)
model.optimize_thresholds(X_val, Y_val)
i = 0
label, res = model.predict_instances(X_val[i], n_tiles=model._guess_n_tiles(X_val[i]))
Y_val_pred, res_val_pred = tuple(zip(*[model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False) for x in X_val[:]]))
plot_metrics(Y_val, Y_val_pred)

i=np.random.randint(len(X_val))
print(i)
plot_img_label_SC(X_val[i],Y_val[i], 1, lbl_title="label GT")
plt.show()
plot_img_label_SC(X_val[i],Y_val_pred[i], 1, lbl_title="label Pred")

#crea_dic_from_merged_masks(masks1,masks2)
masksM=np.load('dati/masks_MC_merged.npy')

cls_dict = crea_dic_from_merged_masks(masksM)
X_val, Y_val, C_val, X_trn, Y_trn, C_trn=split_train_val_MC(tiles,masksM,cls_dict=cls_dict )
model = create_model_auto_grid(tiles, masksM, basedir = 'Models', name = "secondo_modello_multiclasse")
model=train_model(model, X_trn, Y_trn,C_trn,X_val,Y_val,C_val,epochs=200)
Y_val_pred, res_val_pred = tuple(zip(*[model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False) for x in X_val[:]]))
i = 10
plot_img_label_MC(X_val[i],Y_val[i], C_val[i], lbl_title="label GT")
plot_img_label_MC(X_val[i],Y_val_pred[i], class_from_res(res_val_pred[i]), lbl_title="label Pred")

plot_metrics(Y_val, Y_val_pred)

i = 5
plot_img_label_MC(X_val[i],Y_val[i], C_val[i], lbl_title="label GT")
plot_img_label_MC(X_val[i],Y_val_pred[i], class_from_res(res_val_pred[i]), lbl_title="label Pred")

i = np.random.randint(len(X_val))
print(i)
plot_img_label_MC(X_val[i],Y_val[i], C_val[i], lbl_title="label GT")
plot_img_label_MC(X_val[i],Y_val_pred[i], class_from_res(res_val_pred[i]), lbl_title="label Pred")

