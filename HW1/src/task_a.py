import torch

def main():
    # 1. 定义张量 a 和 b，并设置 requires_grad=True 以便追踪梯度
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    # 2. 定义函数 y = a^2 + b^2
    y = a**2 + b**2

    print(f"a = {a.item()}, b = {b.item()}")
    print(f"y = a^2 + b^2 = {y.item()}")

    # 3. 反向传播，自动计算偏导数
    y.backward()

    # 4. 打印梯度结果 (即 y 对 a 和 b 的偏导数)
    # y 对 a 的偏导数是 2a，当 a=2 时，结果应为 4
    print(f"dy/da = {a.grad.item()}")  
    
    # y 对 b 的偏导数是 2b，当 b=3 时，结果应为 6
    print(f"dy/db = {b.grad.item()}")  

if __name__ == "__main__":
    main()
