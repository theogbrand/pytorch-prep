import torch
# automatically stretch tensors to compatible shapes
# Steps:
# 1: Tensors A and B, possibly/not same number of dims. If differnt number of dims, firstly: "Flush all Tensor dims to the right, then "add" "one"-dimensions to the left ONLY
#2: Focusing on the "one"-dims, flip them to match other tensor dim. Repeat for both Tensors A and B
# 3: after "flipping", if Tensor dims ALL match, broadcasting will succeed. 
# Only ones are flexible, everything else must align. BT fails the moment both tensors have different sizes >1, no padding/stretching can help 

# Explanation Source: https://stackoverflow.com/questions/51371070/how-does-pytorch-broadcasting-work/51371509#51371509

# Simple: single number to whole array
scalar = 5.0
tensor = torch.ones(2, 3)
print(scalar + tensor)  # (2,3)
print(torch.add(scalar, tensor))

# Row + Column = Matrix (classic) -> Both tensors are broadcasted
row = torch.tensor([[1, 2, 3]])      # (1,3)
col = torch.tensor([[1], [2], [3]])  # (3,1)
print(row + col)                      # broadcasts to (3,3)

# Application to NLP/Transformers:
hidden = torch.rand(32, 128, 512) # Batch, Time/Sequence Length, Channels/Features/d_model -> Embedding Dims
bias = torch.rand(512)

# Application to CV (Batch, Channels, Height, Width)
t_img = torch.rand(32, 3, 224, 224) # 32 Images, 3 RGB Channels, Height Pixels, Width Pixels