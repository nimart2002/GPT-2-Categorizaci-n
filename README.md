# GPT-2 de categorización: esquemas generativos en un transformer pequeño

Código del TFM *Generative Schemas in LLMs? A project on category induction, episodic retrieval, and consolidation*.

Un transformer de tipo GPT-2 (≈110.000 parámetros), entrenado **desde cero en CPU**, aprende un mundo sintético de categorías. Comparamos sus predicciones con las de un **observador bayesiano exacto** para saber qué partes de la estructura de las categorías ha aprendido realmente.

> **Estado actual:** la fase 1 (adquisición del esquema) está implementada y evaluada. Las fases 2 (memoria episódica) y 3 (réplica y consolidación) están pendientes.

---

## Índice

1. [La pregunta](#1-la-pregunta)
2. [El mundo sintético](#2-el-mundo-sintético)
3. [Qué ve el modelo](#3-qué-ve-el-modelo)
4. [El modelo](#4-el-modelo)
5. [El observador bayesiano](#5-el-observador-bayesiano)
6. [Fases del proyecto](#6-fases-del-proyecto)
7. [Resultados de la fase 1](#7-resultados-de-la-fase-1)
8. [Estructura del repositorio](#8-estructura-del-repositorio)
9. [Instalación y uso](#9-instalación-y-uso)
10. [Configuración](#10-configuración)
11. [Tests](#11-tests)
12. [Supuestos y preguntas abiertas](#12-supuestos-y-preguntas-abiertas)

---

## 1. La pregunta

Saber **clasificar** un objeto no es lo mismo que tener un **concepto** de su categoría. Un concepto generativo permite además:

- **predecir propiedades ocultas** ("no lo veo, pero seguramente tiene…");
- **prever las consecuencias de actuar** ("si hago esto, pasará…");
- **reconocer excepciones** ("este individuo es atípico en…");
- **recordar individuos concretos** sin "normalizarlos" hacia lo típico.

La pregunta del TFM es si un modelo de lenguaje que aprende de registros de experiencia forma **esquemas generativos** de categorías, o solo asocia etiquetas con rasgos superficiales. Después, cómo interactúa ese esquema con la **recuperación episódica** y la **consolidación**, siguiendo el marco de Spens y Burgess (2026).

Para poder contestar, el mundo es **sintético**: conocemos sus ecuaciones y, por tanto, qué significa exactamente "haber entendido una categoría".

---

## 2. El mundo sintético

Implementado en [`minigpt/world.py`](minigpt/world.py) (sección 4 del documento del TFM).

### Variables ocultas

| Variable | Significado | Se sortea |
|---|---|---|
| `u, v` ∈ {0,1} | Categoría c = (u, v): **4 categorías** | una vez por objeto |
| `z` ∈ {0,1} | **Subtipo** dentro de la categoría | una vez por objeto |
| `n1…n5` ~ Bernoulli(η = 0,1) | **Desviaciones individuales** (excepciones estables) | una vez por objeto, fijas |
| `ν1, ν2` ~ Bernoulli(ε) | **Despistes** en los resultados (ruido) | en cada encuentro |

### Propiedades visibles (⊕ = XOR, "es distinto de")

```
F1 = u ⊕ n1        F2 = v ⊕ n2                        rasgos diagnósticos
F3 = z ⊕ n3        F4 = u ⊕ z ⊕ n4    F5 = v ⊕ z ⊕ n5    rasgos relacionales
F6 ~ Bernoulli(½)                                      irrelevante
```

### Acción y resultados

El agente elige una acción `a` ∈ {0,1} y observa:

```
Y1 = u ⊕ z ⊕ a ⊕ ν1
Y2 = v ⊕ Y1 ⊕ ν2
```

### Por qué este diseño

Valores "de libro" (sin desviaciones). Los nombres se sortean en cada mundo; estos son los de `seed=0`:

| Categoría | (u,v) | z | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|---|---|
| RUD | (0,0) | 0 / 1 | 0 | 0 | 0 / 1 | **0 / 1** | 0 / 1 |
| KAV | (0,1) | 0 / 1 | 0 | 1 | 0 / 1 | **0 / 1** | 1 / 0 |
| MIP | (1,0) | 0 / 1 | 1 | 0 | 0 / 1 | **1 / 0** | 0 / 1 |
| ZET | (1,1) | 0 / 1 | 1 | 1 | 0 / 1 | **1 / 0** | 1 / 0 |

- **F1 y F2** delatan la categoría: bastan para clasificar.
- **F3, F4 y F5 valen 0 o 1 al 50 % en todas las categorías**, así que el **prototipo** (la media de cada propiedad) no dice nada sobre ellas.
- Pero su **relación** sí dice mucho: si u = 0, F4 copia a F3; si u = 1, F4 es lo contrario de F3. Por eso:

  **P(F4 = F3 | u = 0) = (1−η)² + η² = 0,82**, mientras que un modelo de prototipo predice **0,5**.

  Esta diferencia es la firma medible de un esquema frente a un prototipo.
- **Los resultados** dependen de la categoría, del subtipo (que nunca se ve) y de la acción. Para predecirlos hay que **inferir z** a partir de F3–F5 y **combinarlo** con u y a: es el conocimiento funcional o generativo.
- **Excepciones (n) frente a despistes (ν).** Una desviación `n` es estable y **debe recordarse** del individuo. Un despiste `ν` no se repite y **no debe memorizarse**. Esta distinción es la base de las fases 2 y 3.

---

## 3. Qué ve el modelo

Implementado en [`minigpt/records.py`](minigpt/records.py) y [`minigpt/dataset.py`](minigpt/dataset.py).

### Formato de registro (21 tokens)

```
CAT MIP F1 1 F2 0 F3 ? F4 1 F5 0 F6 1 ACT 0 Y1 1 Y2 0 END
```

- **`?` = no observado.** El valor se retira de la **entrada**, no solo de la pérdida (§6.4 del documento). `CAT ?` es un registro sin etiqueta.
- **Vocabulario de 20 tokens:** `<pad> <bos> CAT KAV MIP RUD ZET F1…F6 ACT Y1 Y2 0 1 ? END`.
- El modelo **nunca** ve `u, v, z, n, ν` ni las ecuaciones.

### Objetivo de entrenamiento: ecuación (20)

Solo cuentan en la pérdida los **valores** de F1–F6 y de Y1–Y2. La categoría y la acción condicionan la predicción, pero no se predicen; los `?` tampoco. Las posiciones excluidas llevan `-100` en `y`, y `F.cross_entropy(ignore_index=-100)` las ignora.

```
pos  entrada  ->  objetivo
  0    <bos>  ->  (ignorado)
  1      CAT  ->  (ignorado)
  2      RUD  ->  (ignorado)
  3       F1  ->  (ignorado)     F1 oculta
  4        ?  ->  (ignorado)
  5       F2  ->  0              predecir F2 sabiendo la categoría
  ...
 15      ACT  ->  (ignorado)     la acción no se predice
 17       Y1  ->  1              predecir la consecuencia de actuar
```

Como el GPT predice cada token a partir de los anteriores, el orden del registro convierte cada objetivo en una tarea del documento: **predecir una propiedad oculta** (F4 dados la categoría, F1, F2 y F3) o **prever un resultado** (Y1 dados la categoría, las propiedades y la acción).

### Datos

- **Separación por objeto:** 1.000 objetos de entrenamiento y 500 de prueba, sin solapamiento. Generalizar significa responder bien sobre **individuos nunca vistos**.
- **En cada lote:** objetos de entrenamiento sorteados, cada uno con un **encuentro nuevo** (acción y despistes nuevos) y una **máscara de observación nueva** (cada propiedad se oculta con p = 0,3).
- **Un registro por ejemplo** (22 tokens con `<bos>`). En la fase 1 no se juntan registros en el contexto, para medir lo aprendido **en los pesos** y no en contexto.

---

## 4. El modelo

[`minigpt/gpt.py`](minigpt/gpt.py): transformer causal de tipo GPT-2, **inicializado aleatoriamente** (no se cargan pesos preentrenados).

| Parámetro | Valor (§7.3 del documento) |
|---|---|
| Capas | 2 |
| Cabezas de atención | 2 |
| Dimensión (`n_embd`) | 64 |
| Dimensión del MLP | 256 |
| Contexto máximo | 128 |
| Vocabulario | 20 |
| **Parámetros** | **109.568** |

El documento estima unos 116.000 parámetros con un vocabulario de 128 tokens. La diferencia, (128 − 20) × 64 = 6.912, son las filas de embedding que nos ahorramos.

Detalles: LayerNorm *pre-LN*, GELU, *weight tying* entre embedding y salida, inicialización de GPT-2 (σ = 0,02; proyecciones residuales escaladas por 1/√(2·n_layer)). `forward(..., return_hidden=True)` devuelve el flujo residual tras cada bloque, para el análisis de representaciones.

### Entrenamiento ([`minigpt/train.py`](minigpt/train.py))

- AdamW (lr máximo 3·10⁻³, `weight_decay` 0,01), lotes de 64.
- **Calentamiento** lineal de 100 pasos y **decaimiento en coseno** hasta 3·10⁻⁴.
- **Recorte de gradientes** a norma 1.
- Evaluación cada 250 pasos sobre dos conjuntos **fijos** de 2.000 ejemplos (objetos de entrenamiento y objetos de prueba).
- Se guarda el modelo con mejor pérdida de prueba en `checkpoints/fase1.pt`.

---

## 5. El observador bayesiano

[`minigpt/bayes.py`](minigpt/bayes.py) calcula la **mejor predicción posible** para cada token objetivo:

1. Enumera las 8 hipótesis (u, v, z).
2. Pondera cada una por su probabilidad a priori y por la verosimilitud de lo observado. Dado (u, v, z), las propiedades son independientes, así que la verosimilitud es un producto.
3. Predice promediando sobre el posterior.

Respeta la **misma causalidad que el GPT**: para cada posición usa solo lo anterior. Ningún modelo puede tener menor pérdida esperada, así que la distancia al bayesiano mide **qué falta por aprender**. Los tests comparan sus probabilidades con simulaciones del mundo.

---

## 6. Fases del proyecto

| Fase | Pregunta | Mecanismo | Estado |
|---|---|---|---|
| **1. Esquema** | ¿Aprende la estructura de las categorías y la generaliza a objetos nuevos? | Conocimiento **en los pesos** | ✅ Implementada |
| **2. Memoria episódica** | Con un registro previo del mismo objeto en el contexto, ¿recupera sus **excepciones** (n) sin memorizar los **despistes** (ν)? | **Recuperación**: el registro va en el contexto | ⏳ Pendiente |
| **3. Consolidación** | Al reentrenar con registros guardados (réplica), ¿qué se gana en memoria de individuos y qué se pierde en generalización? | **Réplica**: cambian los pesos | ⏳ Pendiente |

Además: análisis de las **representaciones internas** (¿se pueden leer la categoría y el subtipo z de las activaciones?).

---

## 7. Resultados de la fase 1

Todos los resultados son con la configuración por defecto (`world_seed=0`, `seed=0`), en CPU.

### 7.1 Curso del entrenamiento (3.000 pasos, ≈1 min)

```
Modelo: 109,568 parámetros
paso     0 | lr 3.0e-05 | entrenamiento 3.077 (0.0%)  | prueba 3.076 (0.0%)
paso   250 | lr 3.0e-03 | entrenamiento 0.602 (63.3%) | prueba 0.598 (63.4%)
paso   500 | lr 2.9e-03 | entrenamiento 0.597 (64.4%) | prueba 0.593 (64.2%)
paso   750 | lr 2.7e-03 | entrenamiento 0.573 (69.4%) | prueba 0.573 (69.3%)
paso  1000 | lr 2.4e-03 | entrenamiento 0.531 (72.7%) | prueba 0.535 (72.3%)
paso  1500 | lr 1.7e-03 | entrenamiento 0.532 (72.3%) | prueba 0.534 (72.1%)
paso  2000 | lr 1.0e-03 | entrenamiento 0.533 (71.1%) | prueba 0.533 (71.4%)
paso  2500 | lr 4.9e-04 | entrenamiento 0.521 (72.4%) | prueba 0.522 (72.1%)
paso  3000 | lr 3.0e-04 | entrenamiento 0.521 (71.8%) | prueba 0.521 (72.1%)
```

Pérdida media por token objetivo y, entre paréntesis, acierto (argmax). Se distinguen tres etapas:

1. **Inicio (paso 0):** pérdida ≈ ln 20 = 3,00. El modelo reparte la probabilidad entre los 20 tokens.
2. **Regularidades simples (pasos 0–250):** la pérdida cae a ≈ 0,60. El modelo aprende que tras `Fi` viene `0` o `1` y las probabilidades por categoría de F1 y F2.
3. **Estructura relacional (pasos 500–1000):** segunda caída, hasta ≈ 0,53. Aparecen las relaciones entre propiedades (§7.2).

**Entrenamiento ≈ prueba en todo momento:** no hay sobreajuste a individuos; lo aprendido vale para objetos nuevos. El acierto queda en torno al 72 % porque el mundo tiene ruido irreducible (η, ε y F6).

### 7.2 Comparación con el observador bayesiano

[`experiments/paso13_bayes.py`](experiments/paso13_bayes.py), sobre 2.000 ejemplos de **objetos de prueba**. *Exceso* = pérdida GPT − pérdida Bayes. *KL* = KL(Bayes ‖ GPT) media; 0 significa predicciones idénticas.

**3.000 pasos:**

| Campo | Qué exige | GPT | Bayes | Exceso | KL |
|---|---|---|---|---|---|
| F1 | categoría → rasgo | 0,306 | 0,302 | +0,004 | 0,0005 |
| F2 | categoría → rasgo | 0,342 | 0,343 | −0,001 | 0,0007 |
| F3 | nada que predecir (ln 2) | 0,693 | 0,693 | +0,000 | 0,0032 |
| F4 | **u combinado con F3** | 0,542 | 0,541 | +0,001 | 0,0032 |
| F5 | v combinado con el z inferido | 0,542 | 0,501 | +0,041 | 0,0550 |
| F6 | nada que predecir (ln 2) | 0,695 | 0,693 | +0,002 | 0,0012 |
| Y1 | inferir z; u ⊕ z ⊕ a | 0,502 | 0,398 | +0,105 | 0,1155 |
| Y2 | v ⊕ Y1 | 0,533 | 0,310 | +0,223 | 0,2194 |
| **Total** | | **0,519** | **0,461** | **+0,058** | **0,0613** |

**12.000 pasos (≈3,5 min):**

```
paso     0 | entrenamiento 3.077 | prueba 3.076
paso  2000 | entrenamiento 0.600 | prueba 0.591
paso  4000 | entrenamiento 0.500 | prueba 0.501
paso  8000 | entrenamiento 0.495 | prueba 0.499
paso 12000 | entrenamiento 0.493 | prueba 0.494
```

| Campo | GPT | Bayes | Exceso | KL |
|---|---|---|---|---|
| F1 | 0,303 | 0,302 | +0,001 | 0,0004 |
| F2 | 0,343 | 0,343 | +0,000 | 0,0005 |
| F3 | 0,690 | 0,693 | −0,004 | 0,0037 |
| F4 | 0,539 | 0,541 | −0,002 | 0,0029 |
| F5 | 0,516 | 0,501 | +0,015 | 0,0304 |
| F6 | 0,693 | 0,693 | −0,001 | 0,0005 |
| Y1 | 0,476 | 0,398 | +0,079 | 0,0884 |
| Y2 | 0,415 | 0,310 | +0,105 | 0,0978 |
| **Total** | **0,491** | **0,461** | **+0,031** | **0,0344** |

Los excesos ligeramente negativos son ruido de muestreo; la columna KL, que no depende de los valores sorteados, es la medida más fiable.

### 7.3 Interpretación

- **El modelo aprende el esquema relacional de las propiedades.** F4, que exige combinar la categoría con F3 (la relación que distingue un esquema de un prototipo), es **óptima** ya a los 3.000 pasos y en objetos nunca vistos.
- **No se inventa patrones:** en F3 y F6, donde no hay nada que predecir, se queda en ln 2.
- **Las consecuencias de las acciones se aprenden tarde y de forma incompleta.** Y1 requiere inferir el subtipo oculto z y combinarlo con u y a (un XOR de tres variables); Y2 depende de Y1 y v. Cuadruplicar el entrenamiento reduce el exceso total a la mitad, pero Y1 sigue lejos del óptimo.
- **Conclusión provisional:** con este tamaño y presupuesto, el modelo adquiere antes la **estructura entre propiedades** que el **conocimiento funcional** (qué pasa al actuar), que es precisamente lo que el documento considera la marca de un esquema generativo.

### 7.4 Próximos análisis de la fase 1

- Curvas de KL **por campo** a lo largo del entrenamiento (¿en qué orden se aprende cada parte del esquema?).
- Barrido de tamaño del modelo, *learning rate* y duración, para separar capacidad de optimización en Y1/Y2.
- Evaluación **sin etiqueta** (`CAT ?`).
- Varias semillas de mundo y de entrenamiento (barras de error).
- *Probing* de `u`, `v` y `z` en el flujo residual.

---

## 8. Estructura del repositorio

```
minigpt/
  config.py        Configuración: WorldConfig, DataConfig, GPTConfig, TrainConfig
  world.py         El mundo: objetos (variables ocultas + propiedades) y encuentros
  tokenizer.py     Tokenizador genérico (token <-> entero)
  records.py       Formato de registro, vocabulario y máscara de objetivos (ec. 20)
  dataset.py       Separación por objeto, máscaras de observación, lotes con -100
  gpt.py           Modelo GPT (atención causal, MLP, pre-LN, weight tying)
  train.py         Entrenamiento de la fase 1 con evaluación y checkpoints
  bayes.py         Observador bayesiano exacto
experiments/
  paso13_bayes.py  Comparación GPT vs Bayes por campo
tests/             Tests de cada módulo (pytest)
checkpoints/       Modelos entrenados (no se suben a git)
```

---

## 9. Instalación y uso

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Todos los comandos se ejecutan **desde la raíz del repositorio**:

```bash
python -m minigpt.world             # ver objetos y encuentros de ejemplo
python -m minigpt.records           # ver un registro y su máscara de objetivos
python -m minigpt.dataset           # ver un ejemplo de entrenamiento posición a posición
python -m minigpt.bayes             # ejemplos del observador bayesiano (incluye el 0,82)
python -m minigpt.train             # entrenar la fase 1 -> checkpoints/fase1.pt
python -m experiments.paso13_bayes  # comparar el modelo entrenado con Bayes
python -m pytest -q                 # ejecutar los tests
```

---

## 10. Configuración

Todos los valores por defecto están en [`minigpt/config.py`](minigpt/config.py). Para un experimento se cambia solo lo necesario, sin editar ese archivo:

```python
from minigpt.config import DataConfig, TrainConfig
from minigpt.train import train

train(
    DataConfig(p_hide_feature=0.0),
    train_cfg=TrainConfig(steps=12000, out_path="checkpoints/sin_mascara.pt"),
)
```

| Clase | Campos principales (valor por defecto) |
|---|---|
| `WorldConfig` | `eta=(0.1,)*5`, `epsilon=(0.1, 0.1)`, `p_z=0.5`, `p_x6=0.5`, `category_names` |
| `DataConfig` | `n_train_objects=1000`, `n_test_objects=500`, `p_hide_feature=0.3`, `p_hide_category=0.0`, `world_seed=0` |
| `GPTConfig` | `n_layer=2`, `n_head=2`, `n_embd=64`, `ff_dim=256`, `block_size=128`, `dropout=0.0` |
| `TrainConfig` | `steps=3000`, `batch_size=64`, `lr=3e-3`, `min_lr=3e-4`, `warmup_steps=100`, `grad_clip=1.0`, `eval_every=250` |

---

## 11. Tests

35 tests, todos en verde (`python -m pytest -q`). Los más importantes:

| Test | Comprueba |
|---|---|
| `test_world.py::test_relational_feature_0_82` | El mundo reproduce P(F4 = F3 \| u = 0) = 0,82 |
| `test_world.py::test_config_override` | Sin ruido, se cumplen exactamente las ecuaciones |
| `test_records.py::test_hidden_values_are_not_in_the_record` | Un valor oculto no aparece en la entrada |
| `test_records.py::test_target_mask_*` | La máscara implementa la ecuación (20) |
| `test_dataset.py::test_split_by_object` | Entrenamiento y prueba no comparten objetos |
| `test_bayes.py::test_bayes_agrees_with_simulation` | El observador coincide con la simulación del mundo |
| `test_bayes.py::test_target_probs_match_target_mask` | Bayes y la pérdida del GPT evalúan las mismas posiciones |
| `test_gpt.py::test_causal` | El modelo no mira tokens futuros |
| `test_train.py::test_short_training_learns` | Un entrenamiento corto reduce la pérdida y guarda el modelo |

---

## 12. Supuestos y preguntas abiertas

Valores **no especificados** en el documento del TFM (marcados como `SUPUESTO` en `config.py`):

1. **ε = 0,1** para los despistes de Y1 e Y2.
2. **p_hide_feature = 0,3** para las máscaras de observación.

Decisiones pendientes de discutir:

3. **Etiquetas en el entrenamiento.** Ahora `p_hide_category = 0`: el modelo nunca ve `CAT ?` al entrenar, así que evaluar sin etiqueta sería fuera de distribución.
4. **Un solo mundo.** Conviene repetir con varias asignaciones de nombres (`world_seed`).
5. **Tamaño del mundo.** Solo hay 4 × 2⁶ = 256 combinaciones visibles, así que casi todo objeto de prueba tiene un "gemelo" con el mismo vector en el entrenamiento (§5.1 del documento). Hay que precisar qué significa "generalizar a objetos nuevos".
6. **Identificadores de objeto** para la fase 2: cómo tokenizarlos sin que delaten la categoría.

---

*Referencias:* documento del TFM *Generative Schemas in LLMs?* (7 oct. 2026); Spens y Burgess (2026); Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2).
