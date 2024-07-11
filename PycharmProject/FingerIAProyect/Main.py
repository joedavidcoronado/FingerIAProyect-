from Interfaz import Interfaz
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ruta_dataset = r"C:\Users\joedc\Desktop\algoritmica\fingers\ok"

    # Crear instancia del clasificador de gestos
    clasificador = Interfaz()

    # Cargar y dividir el dataset
    X, y = clasificador.cargar_dataset(ruta_dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    clasificador.entrenar_modelo(X_train, y_train)

    # Evaluar el modelo
    precision = clasificador.evaluar_modelo(X_test, y_test)
    print(f'Precisión del modelo: {precision:.2f}')

    # Iniciar la interfaz gráfica
    clasificador.iniciar_interfaz()

