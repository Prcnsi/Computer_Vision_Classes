import torch
import torch.nn as nn

# tensor란 다차원 배열을 나타내는 자료형, GPU를 이용한 연산을 지원한다. 리스트나 넘파이 배열은 tensor로 변경 가능. (datatype도 그냥 torch.자료형[int, float, ~]으로 지정)
# torch.Tensor()는 주로 텐서의 구조를 미리 정의하고 나중에 값을 채울 때 사용(기본 float32), torch.tensor()는 데이터를 기반으로 즉시 텐서를 생성할 때 사용 (tensor, Tensor는 기본 데이터타입이 다름.)
data = [[1, 2, 3], [4, 5, 6]]
x=torch.tensor(data, dtype=torch.float64)
print("tensor x:",x)

# Pytorch는 연산을 실행할 장치를 지정 가능
device = torch.device('cpu')

# torch.zeros((shape)) | torch.ones((shape)) | torch.rand((shape))로 텐서 생성 가능. 
rand_tensor=torch.rand((4,3),device=device) 

# rand 함수는 0과 1사이의 정규분포(균등 분포)에서 난수 생성
print("rand_tensor:", rand_tensor)

# 대문자 .Tensor는 데이터를 넣어도 되고, shape을 입력해 텐서 생성 가능 
a=torch.Tensor([1,2,3])
rand_a=torch.Tensor(3,3)
print(a)
print(rand_a)

# .Tensor나 .empty 함수는 메모리 상에 할당된 임의의 값으로 초기화된 텐서를 생성
empty_tensor=torch.empty(3,4)
print("empty tensor:", empty_tensor)

# .dtype는 데이터타입과 .numel()은 크기, .size() 모양, .view()모양 변경 (pytorch에서 view==reshape)
test_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
print("numel, size, shape, ", test_tensor.numel(), test_tensor.size(), test_tensor.shape)  # torch.Size([3, 2]) torch.Size([3, 2])
print("shape 변경 후:", test_tensor.view(1,6),'\n')

#================================================================================================================
# nn.Linear(input_dimension, output_dimentsion)
linear=nn.Linear(10,20)

input_tensor=torch.randn(1,10)
output_tensor=linear(input_tensor)

print(input_tensor, '\n', output_tensor)

# tensor의 인덱싱과 슬라이싱
my_tensor=torch.tensor([[1,2,3],
                        [4,5,6],
                        [7,8,9]])

# tensor[행][열의 조건 True/False], tensor[행, 열], tensor[:,열]-모든 행에서 특정 열, tensor[행,:]-특정 행에서 모든 열, tensor[행]-원하는 행
print(my_tensor[1])
print(my_tensor[2])
print(my_tensor[:,0])
print(my_tensor[1,:])
print(my_tensor[2,2])

print(my_tensor[my_tensor==3])

# inplace 연산은 대체하는 것으로 지정된 크기와 값으로 채울 수 있음. 
x = torch.zeros(3, 3)

# 모든 요소를 1로 채우기
x.fill_(1)
print(x)

# 3x3 크기의 텐서를 5로 채우기, torch.full((shape),inplace_values)
x2 = torch.full((3, 3), 5)
print(x2)