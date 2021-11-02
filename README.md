# overlapsAndGroupings
 A project that tries to leverage overlap and grouping functions inside CNNs. Documentation available in English and Spanish.

* [Principal use cases (EN)](#principal-use-cases-en).
* [Usos principales (ES)](#usos-principales-es)
  * [Método de funcionamiento del script de pruebas](#metodo-de-funcionamiento-del-script-de-pruebas)
  * [Generación de hojas de cálculo con resultados](#generacion-de-hojas-de-calculo-con-resultados)
  * [Organización de archivos de log para análisis mediante Tensorboard](#organizacion-de-archivos-de-log-para-analisis-mediante-tensorboard)

## Principal use cases (EN)

Coming soon...


## Usos principales (ES)

### Método de funcionamiento del script de pruebas

1. El punto de entrada del proceso de entrenamiento es el script /src/runs/run_test.py

    Este script lleva a cabo los siguientes procesos:
	1. Lee múltiples parámetros de entrada mediante el método argparse() que especifican las características de la prueba a realizar. Se parsean y se guardan sus valores como variables.
    
        Nota: En la carpeta /runs se provee un bash script de muestra sobre un posible uso del proyecto.  
	2. Se llama a la función full_test(), enviándole los parámetros leídos previamente. El resto de pasos se realizan en la función full_test().
	3. Se leen el resto de parámetros necesarios a partir de una archivo .json situado en la carpeta /config. Los parámetros de la prueba se almacenan en un diccionario "info_data" que se usará posteriormente.
	4. Se crean carpetas para los resultados del test en las carpetas /reports/models, /reports/results y /reports/runs.
		
        Nota: 
        
            /reports/models: Almacena los parámetros de los modelos ya entrenados para su posterior carga (si fuera necesario).  
            /reports/results: Almacena los resultados de métricas de los distintos modelos entrenados, medidos sobre la partición de test (en ficheros de nombre test_metrics.txt).
		 	/reports/runs: Almacena logs generados usando la herramienta Tensorboard para su posterior carga con dicha herramienta.
	5. Se entra en el bucle principal del programa: Cada iteración del bucle entrena un modelo distinto y repite los siguientes pasos:
        1. Se genera una nueva carpeta dentro de la carpeta de test creada previamente con nombre "test0", "test1", etc.
      	2. Se genera un nuevo modelo a partir de los parámetros cargados. Se generan los objetos optimizer, scheduler y criterion necesarios para el entrenamiento.
		3. Se llama al método train() del script /src/model_tools/train.py -> Ver parte 2
		4. Tras la ejecución del método train(), se obtiene un objeto model que incluirá el modelo entrenado sobre la partición de train del dataset indicado.
		5. Se llama al método get_prediction_metrics() del script /src/model_tools/evaluate.py con el objeto model y la partición de test, y posteriormente al método log_eval_metrics() del script /src/data/save_results.py para generar un fichero de nombre "test_metrics.txt" con los resultados de la evaluación.
		6. Se llama al método save_model() del script /src/model_tools/save_model.py con el objeto model y el diccionario de valores "info_data" cargado en el punto 1.3, que facilitará la carga de datos.
		7. 
2. Proceso de entrenamiento de train()
	
    Este método es el encargado de llevar a cabo el entrenamiento de un modelo dado. 
	
    Recibe como entrada los siguientes parámetros (Nota: no todos los parámetros son obligatorios):
		
        * name: Nombre del test (generado en la función full_test())
		* model: Modelo a entrenar.
		* optimizer, criterion, scheduler: Objetos que guían el proceso de aprendizaje del modelo.
		* train_loader, val_loader: Dataloaders que cargan los datos de entrenamiento y validación, respectivamente
		* using_tensorboard: Si es True, genera logs que posteriormente permitirán visualizar el entrenamiento usando la herramienta Tensorboard. NOTA: De aquí en adelante asumimos que es True
		*otros
	1. Se comienza consultando si se dispone de un dispositivo CUDA desde el que llevar a cabo la ejecución de las pruebas.
	2. Se genera un archivo de log en la carpeta /reports/runs/name (donde "name" hace referencia al nombre de nuestro test) para ser leído usando la herramienta Tensorboard. Se genera un objeto SummaryWriter que será el encargado de escribir este log.
	3. Se realiza una primera escritura de los atributos a ser guardados por el modelo mediante el objeto SummaryWriter.
	4. Se entra en el bucle de entrenamiento (se repetirá tantas veces como establezca el atributo "num_epochs"):
		1. Se inicia un nuevo bucle que recorre todos los batches de la partición de entrenamiento (se repetirá tantas veces como batches haya en train_val)
			1. El entrenamiento del modelo se lleva a cabo aquí.
			2. Cada "iters_per_log" iteraciones se loggean nuevos resultados mediante el objeto SummaryWriter (referentes a la partición de entrenamiento)
		2. Tras el bucle de entrenamiento se loggean nuevos resultados mediante el objeto SummaryWriter (referentes a la partición de validación)
	5. Se devuelve el modelo entrenado

### Generación de hojas de cálculo con resultados

Nota: Requiere haber almacenado los resultados de la evaluación de distintos modelos en una carpeta "group_of_tests" con la estructura:
	
    /reports
		/results
			/group_of_tests
				/test_name0
					/test0
						test_metrics.txt
					/test1
						test_metrics.txt
					...
				/test_name1
					/test0
						test_metrics.txt
					/test1
						test_metrics.txt
					...
				...

Donde "test_name1", "test_name2", etc. son nombres de ejemplo para una serie de tests ejecutado mediante el script /src/runs/run_test.py y "group_of_tests" es el nombre de una carpeta que los contiene a todos ellos (generada manualmente por el usuario).

Basta con ejecutar el script /src/data/summarize_excel.py indicando como parámetro el nombre de la carpeta padre (en el ejemplo, "group_of_tests").
	
1. Se cargan los parámetros de entrada mediante la función argparse() y se almacenan en variables.
2. Se llama a la función summarize_experiments() enviándole los parámetros cargados previamente. El resto de pasos se realizan en la función summarize_experiments().
3. Se obtiene el nombre de todas las carpetas presentes dentro de la carpeta /reports/results/group_of_tests
	1. Se obtiene el nombre de todas las carpetas presentes dentro de la carpeta de cada test
		1. Se recorren todas las carpetas "test0", "test1", etc. leyendo el contenido de los ficheros "test_metrics.txt" de cada una de ellas (Nota: Solo se lee la información referente a la accuracy obtenida).
4. Se genera un DataFrame de pandas con las métricas leídas. Se calculan la media y desviación estándar de cada prueba "test_name0", "test_name1", etc.
5. Se escribe el contenido del DataFrame a un archivo .xlsx.


### Organización de archivos de log para análisis mediante Tensorboard

Nota: Requiere haber almacenado los logs del entrenamiento de distintos modelos en una carpeta "group_of_tests" con la estructura:
	
    /reports
		/results
			/group_of_tests
				/test_name0
					/test0
						test_metrics.txt
					/test1
						test_metrics.txt
					...
				/test_name1
					/test0
						test_metrics.txt
					/test1
						test_metrics.txt
					...
				...

Donde "test_name1", "test_name2", etc. son nombres de ejemplo para una serie de tests ejecutado mediante el script /src/runs/run_test.py y "group_of_tests" es el nombre de una carpeta que los contiene a todos ellos (generada manualmente por el usuario).

Basta con ejecutar el script /src/data/organize_runs.py indicando como parámetro el nombre de la carpeta padre (en el ejemplo, "group_of_tests").

1. Se cargan los parámetros de entrada mediante la función argparse() y se almacenan en variables.
2. Se llama a la función organize_experiments() enviándole los parámetros cargados previamente. El resto de pasos se realizan en la función organize_experiments().
3. Se crea una nueva carpeta /reports/results/group_of_tests/tests
4. Se obtiene el nombre de todas las carpetas presentes dentro de la carpeta /reports/results/group_of_tests
	1. Se obtiene el nombre de todas las carpetas presentes dentro de la carpeta de cada test
		1. Se recorren todas las carpetas "test0", "test1", etc. copiándolas, junto con sus contenidos, a nuevas carpetas con nombre "/reports/results/group_of_tests/tests/test_name0_test_0", etc.
	
Posteriormente, puede ejecutarse la herramienta tensorboard, desde la carpeta /reports/results/group_of_tests mediante el comando "tensorboard --logdir tests".
