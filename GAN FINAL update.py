#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


# In[19]:


# Load data
Bearing_Data_Raw = pd.read_excel('Bearing Data.xls',index_col=-2)
Bearing_Data_Raw['Misaligned'] = np.where((Bearing_Data_Raw['PSIx th']<=0.05) & (Bearing_Data_Raw['PSIy th']<=0.05), 0, 1)
Bearing_Data = Bearing_Data_Raw[['f','e','hmin0','hminL','PSIx th','PSIy th','D0 th','Misaligned']]
Bearing_Data['Misaligned'].value_counts()


# In[147]:


# Separate features and target variable
X = Bearing_Data.iloc[:, :-1]
Y = Bearing_Data .iloc[:, -1]


# In[148]:


# Perform SMOTE
smote = SMOTE(sampling_strategy=0.6)
X_smote, Y_smote = smote.fit_resample(X, Y)
X_smote = X_smote.reset_index(drop=True)


# In[149]:


Bearing_Data_Resampled = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(Y_smote)], axis=1)
Bearing_Data_Resampled['Misaligned'].value_counts()


# In[74]:


sns.countplot(x='Misaligned', data=Bearing_Data)
plt.suptitle("Aligned - Misaligned cases of Original Data")
plt.show()


# In[75]:


sns.countplot(x='Misaligned', data=Bearing_Data_Resampled)
plt.suptitle("Aligned - Misaligned cases of Resampled Data")
plt.show()


# In[70]:


sns.set(style='ticks')
sns.pairplot(
    Bearing_Data,
    x_vars=["f", "e", "hmin0","hminL"],
    y_vars=["f", "e", "hmin0","hminL"], hue='Misaligned'
)
plt.suptitle("Distribution of Original Data",y=1.05)
plt.show()


# In[72]:


sns.pairplot(
    Bearing_Data_Resampled,
    x_vars=["f", "e", "hmin0","hminL"],
    y_vars=["f", "e", "hmin0","hminL"], hue='Misaligned'
)
plt.suptitle("Distribution of Resampled Data",y=1.05)
plt.show()


# In[150]:


# Define the resampled data for the GAN
df = Bearing_Data_Resampled
df


# In[151]:


target_col = "target"
target = df.pop('Misaligned')


# In[152]:


data = df.values


# In[153]:


# Define the generator network
def generator_network():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(7,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(7, activation="linear"))
    return model


# In[154]:


# Define the discriminator network
def discriminator_network():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(7,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


# In[155]:


def get_real_data(data, target, batch_size):
    indexes = np.random.choice(data.shape[0], batch_size, replace=False)
    return data[indexes], target[indexes]


# In[156]:


# Compile the generator network
generator = generator_network()
generator.compile(loss="binary_crossentropy", optimizer="adam")


# In[157]:


# Compile the discriminator network
discriminator = discriminator_network()
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[158]:


# Combine the generator and discriminator into a GAN
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer="adam")


# In[161]:


# Train the GAN
epochs = 1000
batch_size = 128

for i in range(epochs):
    # Train the discriminator
    X, y = get_real_data(data, target, batch_size)
    d_loss = discriminator.train_on_batch(X, y)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 7))
    y_gen = np.ones(batch_size)
    g_loss = gan.train_on_batch(noise, y_gen)
    
    print("i: ", i)
    print("epochs: ", epochs)
    print("d_loss: ", d_loss)
    print("g_loss: ", g_loss)
    print("y_gen: ", y_gen)
    
    d_loss_str = [str(x) for x in d_loss]
    print("Epoch {}/{}, D Loss: {}, G Loss: {:.4f}, y_gen: {}".format(i+1, epochs, d_loss_str, g_loss, str(y_gen)))


# In[162]:


# Generate new data using the generator network
noise = np.random.normal(0, 1, (data.shape[0], 7))
generated_data = generator.predict(noise)


# In[163]:


# Concatenate the generated data with the original data
expanded_data = np.concatenate([data, generated_data])


# In[164]:


# Repeat the process 4 more times to get 5 times the original size of the dataset
for i in range(4):
    noise = np.random.normal(0, 1, (data.shape[0], 7))
    generated_data = generator.predict(noise)
    expanded_data = np.concatenate([expanded_data, generated_data])


# In[165]:


# Create a new dataframe from the expanded data
expanded_df = pd.DataFrame(expanded_data)


# In[171]:


# Define your expanded data
expanded_data = np.array(expanded_data)
Bearing_extended_data = pd.DataFrame(expanded_data, columns=['f','e','hmin0','hminL','PSIx th','PSIy th','D0 th'])
Bearing_extended_data['Misaligned'] = np.where((Bearing_extended_data['PSIx th']<=0.05) & (Bearing_extended_data['PSIy th']<=0.05), 0, 1)
Bearing_extended_data


# In[168]:


# Save the expanded data as a csv file
np.savetxt("Bearing_Expanded_Dataset.csv", expanded_data, delimiter=",")


# In[172]:


Bearing_extended_data['Misaligned'].value_counts()


# In[175]:


sns.countplot(x='Misaligned', data=Bearing_extended_data)
plt.suptitle("Aligned - Misaligned cases of Expanded Data")
plt.show()


# In[176]:


sns.pairplot(
    Bearing_extended_data,
    x_vars=["f", "e", "hmin0","hminL"],
    y_vars=["f", "e", "hmin0","hminL"], hue='Misaligned'
)
plt.suptitle("Distribution of Resampled Data",y=1.05)
plt.show()


# In[ ]:




