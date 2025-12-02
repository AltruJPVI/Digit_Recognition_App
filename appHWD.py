from torchvision import transforms
import torch
import streamlit as st
import streamlit_drawable_canvas as draw
from PIL import Image
import matplotlib.pyplot as plt
from ArquitecturaHWD import CNN

def load_model():
    model = CNN()
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Función para predecir el número
def predict_digit(image):
    # Usar la misma transformación que en el entrenamiento
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # Convierte a tensor y escala a [0, 1]
    ])
    
    image = transform(image).unsqueeze(0)
    
    # Guardamos la imagen procesada para visualización
    processed_img = image.clone().squeeze(0).numpy()
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    top_probs, top_labels = torch.topk(probabilities, 3)  # Top 3 predicciones
    return top_labels.numpy()[0], top_probs.numpy()[0], processed_img

# Interfaz con Streamlit
st.title("Reconocimiento de Dígitos Escritos a Mano")
st.write("Dibuja un número en el canvas y haz clic en 'Predecir' para ver los resultados.")

# Dividir la pantalla en dos columnas
col1, col2 = st.columns([2, 1])

# Canvas de dibujo con trazo más delgado
with col1:
    canvas_result = draw.st_canvas(
        fill_color="black",
        stroke_width=15,  # Reducido de 20 a 15 para mejor precisión
        stroke_color="white",
        background_color="black",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas"
    )

# Botón para predecir
if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Convertir canvas a imagen y preprocesar
        image = Image.fromarray(canvas_result.image_data[:, :, :3])  # Ignorar canal alfa
        
        # Predicción
        labels, probs, processed_img = predict_digit(image)
        
        # Visualización de la imagen procesada
        with col2:
            st.write("### Imagen Procesada:")
            
            # Convertir la imagen procesada a formato adecuado para visualización
            # Desnormalizar la imagen
            processed_img = processed_img * 0.5 + 0.5  # Desnormalizar de [-1, 1] a [0, 1]
            
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(processed_img[0], cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            
            st.write("### Predicciones:")
            for i, (label, prob) in enumerate(zip(labels, probs)):
                if i == 0:
                    st.success(f"{label}: {prob*100:.2f}%")  # Predicción principal
                else:
                    st.info(f"{label}: {prob*100:.2f}%")  # Predicciones secundarias
    else:
        st.warning("Por favor, dibuja un número en el canvas antes de hacer clic en 'Predecir'.")

# Agregar consejos para mejorar el reconocimiento
st.markdown("""
### Tips para mejor reconocimiento:
- Intenta centrar el dígito en el canvas
- Escribe números claros y completos
- El trazo debe ser continuo y visible
- Evita números muy pequeños o muy grandes
""")