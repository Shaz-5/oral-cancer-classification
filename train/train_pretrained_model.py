import os
import pathlib
import shutil
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers
from tensorflow.keras.optimizers import Adam

def get_file_names(category, original_dir):
    return list((pathlib.Path(original_dir) / category).glob("*.jpg"))  # Adjust extension as needed

def make_subset(split_name, files_dict, output_dir):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for category, files in files_dict.items():
        category_dir = split_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy(file, category_dir)

def load_dataset(path, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=batch_size
    )
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))

def create_and_train_model(train_dataset, val_dataset, model_name, 
                           input_shape=(224, 224, 3), num_classes=3, 
                           learning_rate=0.0005, l2_reg=0.005, 
                           dropout_rate=0.5, epochs=20):

    base_model_class = getattr(keras.applications, model_name)
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(units=100, activation='relu', 
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(rate=dropout_rate)(x)

    predictions = layers.Dense(units=num_classes, activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    callbacks = [keras.callbacks.ModelCheckpoint(filepath=f"{model_name.lower()}.keras", 
                                                 save_best_only=True, monitor="val_loss")]

    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)
    
    return model

def main(args):
    if (args.output_dir):
        output_dir = pathlib.Path(args.output_dir)

    # step 1: train test val split
    if args.original_dir and output_dir:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        categories = ["normal", "pre_cancer", "oral_cancer"]
        train_files, val_files, test_files = {}, {}, {}

        for category in categories:
            all_files = get_file_names(category, args.original_dir)
            train, temp = train_test_split(all_files, test_size=args.split_ratio[0], random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)

            train_files[category] = train
            val_files[category] = val
            test_files[category] = test

        make_subset("train", train_files, output_dir)
        make_subset("val", val_files, output_dir)
        make_subset("test", test_files, output_dir)

        print('Created training, validation, and testing subsets.')

        train_dataset_path = output_dir / 'train'
        val_dataset_path = output_dir / 'val'
        test_dataset_path = output_dir / 'test'
    else:
        train_dataset_path = pathlib.Path(args.dataset_dir) / 'train'
        val_dataset_path = pathlib.Path(args.dataset_dir) / 'val'
        test_dataset_path = pathlib.Path(args.dataset_dir) / 'test'

    # step 2: load datasets
    batch_size = args.batch_size
    train_dataset = load_dataset(train_dataset_path, batch_size)
    val_dataset = load_dataset(val_dataset_path, batch_size)
    test_dataset = load_dataset(test_dataset_path, batch_size)

    # step 3: train model
    model = create_and_train_model(train_dataset, val_dataset, args.model_name,
                                   input_shape=(224, 224, 3), num_classes=args.classes,
                                   learning_rate=args.learning_rate, l2_reg=args.l2_reg,
                                   dropout_rate=args.dropout_rate, epochs=args.epochs)

    # step 4: evaluate model
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a pretrained model on a dataset.')

    # required arguments
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name from Keras applications.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs.')

    # optional arguments
    parser.add_argument('--original_dir', type=str, help='Original dataset directory to split.')
    parser.add_argument('--output_dir', type=str, help='Output directory for split datasets.')
    parser.add_argument('--dataset_dir', type=str, help='Directory to load data from if not splitting.')
    parser.add_argument('--split_ratio', type=float, nargs=2, default=[0.3, 0.5], help='Split ratios for train, validation, and test.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation.')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer.')
    parser.add_argument('--l2_reg', type=float, default=0.005, help='L2 regularization factor.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the model.')
    parser.add_argument('--classes', type=int, default=3, help='No. of classes')

    args = parser.parse_args()

    if args.dataset_dir:
        if not args.original_dir and not args.output_dir:
            print("If --dataset_dir is provided, --original_dir and --output_dir are not required.")
    else:
        if not args.original_dir or not args.output_dir:
            parser.error("Both --original_dir and --output_dir must be provided if --dataset_dir is not specified.")

    main(args)
