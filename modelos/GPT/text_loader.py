class TextLoader:
    def __init__(self, data, context_size, batch_size, device):
        self.data = data
        self.context_size = context_size
        self.batch_size = batch_size
        self.pointer = 0
        self.num_batches = len(data) // (context_size * batch_size)
        
        self.device = device

    def __len__(self):
        return self.num_batches

    def get_batch(self):
        # Calculate the start and end of the batch based on the pointer
        start = self.pointer * self.context_size * self.batch_size
        end = start + self.context_size * self.batch_size

        # Wrap around if the pointer exceeds the length of data
        if end > len(self.data):
            self.pointer = 0
            start = 0
            end = self.context_size * self.batch_size

        # Prepare the input and target tensors
        x = self.data[start:end].view(self.batch_size, self.context_size).to(self.device)
        y = self.data[start + 1 : end + 1].view(self.batch_size, self.context_size).to(self.device)

        # Move the pointer forward
        self.pointer += 1

        return x, y

    def reset(self):
        self.pointer = 0
