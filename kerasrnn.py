import os
import pandas as pd
import tensorflow as tf
from itertools import chain
from keras.layers import Embedding, Dense, LSTM
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

folder = './current_corpora/'
lex_entropy_hist_df = pd.DataFrame(columns=['corpus', 'epoch', 'loss', 'acc', 'val_loss', 'val_acc'])
count = 0

if __name__ == '__main__':
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            count += 1
            print(f'Starting {filename}...{count}/22')
            corpus = os.path.splitext(os.path.basename(filename))[0].split('_')[0]
            text = open(os.path.join(folder, filename), 'r', encoding='utf-8').read().splitlines()
            chars = list(set(chain(*(char for line in text for char in line if line if not char.isspace()))))
            tokens = list(set(chain(*(line.split() for line in text if line))))
            chars.insert(0, '<')
            chars.insert(1, '>')
            vocab_size = len(chars)
            max_len = len(max(tokens, key=len))

            ch2idx = {c: i for i, c in enumerate(chars)}
            idx2ch = {i: c for i, c in enumerate(chars)}
            sequences = [chars[0] + t + chars[1] for t in tokens]

            X = pad_sequences([[ch2idx[i] for i in seq[:-1]] for seq in sequences])
            y = pad_sequences([[ch2idx[i] for i in seq[1:]] for seq in sequences])
            split_index = int(0.8 * len(X))
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]
            
            batch_size = 32
            num_epochs = 100
            loss_function = SparseCategoricalCrossentropy()
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            optimizer = Adam()

            model = Sequential()
            model.add(Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True))
            model.add(LSTM(128, return_sequences=True))
            model.add(Dense(vocab_size, activation='softmax'))
            model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[callback], validation_data=(X_val, y_val), verbose=2)
            temp = pd.DataFrame({
                'corpus': corpus, 
                'epoch': len(history.history['loss']),
                'loss': history.history['loss'][-1],
                'acc': history.history['accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_acc': history.history['val_accuracy'][-1]
                }, index=[0]
            )
            lex_entropy_hist_df = pd.concat([lex_entropy_hist_df, temp], ignore_index=True)
            
    with open('lex_entropy_hist_df.csv', mode='w') as f:
        lex_entropy_hist_df.to_csv(f)
            