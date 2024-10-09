# Basado en el código original de qianyingcao (https://github.com/qianyingcao/Laplace-Neural-Operator)
# Licensed under the MIT License

'''
Author: Erik Jon Pérez Mardaras
email: erikjon.perez@gmail.com
LinkedIn: https://www.linkedin.com/in/erikjon-perez-mardaras/
Github: https://github.com/erikjonperez

September 2024
'''


#testing------------------------------------------------------------------------
#User input:
example_index=127
tensor_in=x_test
path_cargado="./lno_model/pesitos.pth"
width=8
modes=32


modelito = LNO1d(width, modes)
modelito.load_state_dict(torch.load(path_cargado))
modelito = modelito.cuda()  
modelito.eval() 
with torch.no_grad():
    x_new = torch.tensor(tensor_in)  
    x_new = x_new.cuda()  
    start_time = time.time()  
    output = modelito(x_new)   
    end_time = time.time()    
    # Calculate inference time
    inference_time = end_time - start_time
    print(f"Tiempo de inferencia: {inference_time:.6f} segundos")
torch.cuda.empty_cache()
y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
output_np = output.cpu().numpy() if output.is_cuda else output.numpy()
output_np = output_np.squeeze(axis=-1) 
y_test_example = y_test_np[example_index]
output_example = output_np[example_index]

#graphic
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test_example, label='Ground Truth data', color='blue', linestyle='-', marker='o', markersize=2)
ax.plot(output_example, label='LNO predicted', color='red', linestyle='--', marker='x', markersize=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
ax.set_title(f'Ground-truth data vs. LNO predicted points for {example_index}')
ax.legend()
file_name = f'Test_with_index_{example_index}.png'
file_path = os.path.join("./lno_model/", file_name)
fig.savefig(file_path)
plt.show()


#metric calculation-------------------------------------------------------------------
#MAE
mae = mean_absolute_error(y_test_example, output_example)
print(f'MAE (Mean Absolute Error): {mae:.6f}')
#MSE
mse = mean_squared_error(y_test_example, output_example)
print(f'MSE (Mean Squared Error): {mse:.6f}')
#RMSE
rmse = np.sqrt(mse)
print(f'RMSE (Root Mean Squared Error): {rmse:.6f}')
#R²
r2 = r2_score(y_test_example, output_example)
print(f'R² (R-squared): {r2:.6f}')
#Inference time:
print(f'Tiempo de la inferencia (s):', inference_time)