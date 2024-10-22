class TextLoader:
    def __init__(self, data, context_size, batch_size, device):
        self.data = data
        self.context_size = context_size
        self.batch_size = batch_size
        self.pointer = 0
        self.num_batches = len(data) // (context_size * batch_size) + 1
        
        self.device = device

    def __len__(self):
        return self.num_batches

    def get_batch(self):
        start = self.pointer * self.context_size * self.batch_size
        end = start + self.context_size * self.batch_size

        if end > len(self.data):
            self.pointer = 0
            start = 0
            end = self.context_size * self.batch_size

        x = self.data[start:end].view(self.batch_size, self.context_size).to(self.device)
        y = self.data[start + 1 : end + 1].view(self.batch_size, self.context_size).to(self.device)

        self.pointer += 1

        return x, y

    def reset(self):
        self.pointer = 0
