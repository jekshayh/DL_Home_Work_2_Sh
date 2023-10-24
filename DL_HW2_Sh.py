import torch
import pandas
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(2023)

class TitanicDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pandas.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        self.df['Age'].fillna(self.df['Age'].median(), inplace=True) #1 Меняем пропущеные значения на медиану
        self.df = self.df.dropna()
        self.df = pandas.get_dummies(self.df, columns=['Sex', 'Pclass']) #2 Добавляем в модель категорийные переменные

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        alive = torch.Tensor([1, 0])
        dead = torch.Tensor([0, 1])
        y = alive if row['Survived'] else dead
        x = torch.Tensor([row['Age'], row['Fare'], row['SibSp'], row['Sex_female'], row['Sex_male'], row['Pclass_1'], row['Pclass_2'], row['Pclass_3']]) #2 Добавляем в модель категорийные переменные
        return x, y


titanic_dataset = TitanicDataset()
dataloader = DataLoader(dataset=titanic_dataset, batch_size=2, shuffle=True)

class SurvivalPredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x

model = SurvivalPredictorPerceptron(input_size=8, hidden_size=1000, output_size=2) #3
optimizer = torch.optim.SGD(model.parameters(), lr=2e-6) #3
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 200

for epoch in range(num_epochs):
    error = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error += loss

        loss.backward()
        optimizer.step()

    print(error/len(titanic_dataset))