import onnx

model = onnx.load("tmp/dynamo_const_nodyn_fix.onnx")
g = model.graph

print("== inputs ==")
for inp in g.input:
    print(" ", inp.name)

print("== initializers ==")
for init in g.initializer:
    print(" ", init.name, init.dims, init.data_type)

# 入力と initializer の交差（initializer なのに input にも出ていると ORT 的には override 可能扱い）
input_names = {i.name for i in g.input}
init_names = {i.name for i in g.initializer}
print("== in both input and initializer ==")
print(input_names & init_names)
