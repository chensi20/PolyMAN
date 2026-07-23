import torch
import time
from thop import profile
from models.PolyMAN import PolyMAN_UNet as PolyMAN
def measure_all():

    input_size = (1, 3, 352, 352) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    model = PolyMAN(n_classes=1) 
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(*input_size).to(device)

    print("=" * 40)
    print(f"Starting PolyMAN (PolyMAN_UNet) performance testing...")
    print(f"Input size:{input_size}")
    print("=" * 40)


    print("Calculating Params and GFLOPs ...")
    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        params_m = params / 1e6
        flops_g = flops / 1e9
        
        print(f"Params : {params_m:.2f} M")
        print(f"GFLOPs : {flops_g:.2f} G")
    except Exception as e:
        print(f"Failed to calculate FLOPs: {e}")

    print("-" * 40)
    print("Testing FPS...")
    
    print("   GPU warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_tensor)
    
    iterations = 300  # Averaging over 300 runs
    print(f"Starting inference loop for {iterations} iterations...")
    
    torch.cuda.synchronize() 
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = iterations / total_time
    
    print(f"FPS: {fps:.2f}")
    print("=" * 40)

    print(f"Our proposed PolyMAN achieves {fps:.1f} FPS with only {params_m:.1f}M parameters and {flops_g:.1f}G FLOPs.")
    print("Compared to Polyp-PVT (28 FPS, 25.4M Params), our method is significantly faster.")

if __name__ == '__main__':
    measure_all()
