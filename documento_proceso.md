# Asistente Conversacional sobre la Ley de Contrato de Trabajo N° 20.744

**Challenge Final — Get Talent**
**Alumno:** Cristian Herrera

---

## Por qué elegí este tema

Cuando tuve que elegir un tema para el
proyecto final, pensé en varios temas pero el otro dia se me aparecio un blog sobre la Ley del Contrato de Trabajo entonces la elegí y me gusto porque es un texto largo, con
estructura clara (artículos numerados, agrupados en títulos y capítulos), y
porque además fue reformada hace muy poco (Ley 27.802, marzo de 2026), así
que el sistema iba a estar respondiendo sobre algo vigente y actual, no un
texto viejo que ya todo el mundo usó de ejemplo.

También elegí este tema porque me permitía usar la metadata de la base de datos
vectorial de una forma que tuviera sentido real: cada artículo con su
título, su capítulo y si está derogado, no como un campo decorativo sino
como algo que efectivamente cambia la respuesta.

---

## Cómo armé la base de conocimiento

Esta fue la parte que más tiempo me llevó, y donde más aprendí sobre lo
que significa realmente "preparar datos" para un sistema de este tipo.

### Descargar y limpiar el texto

Bajé el texto de la ley desde el sitio oficial (argentina.gob.ar). La
primera vez que lo descargué y revisé el reporte, tenía 182.887 caracteres
—de sobra sobre el mínimo de 100.000 que pedía la consigna—, pero al mirar
el final del archivo me di cuenta de que traía pegado un anexo de
"Antecedentes normativos": una lista de qué artículo fue modificado por qué
ley, que no es parte del articulado en sí. Tuve que ajustar el script de
limpieza para cortar ese anexo sin llevarme puesto contenido real. Aprendí
que descargar el texto es solo el primer paso: hay que revisarlo con
cuidado antes de asumir que está listo para usar.

### Partir el texto en fragmentos (chunking)

Acá tuve que pensar distinto a como lo había hecho en el challenge anterior
del bootcamp, donde partía el texto cada 500 caracteres fijos. Con una ley
eso no funciona bien, porque muchos artículos tienen una nota al final que
dice si fueron modificados o derogados, y si el corte cae justo en el medio,
se puede perder esa información o quedar separada del artículo al que
corresponde.

Entonces decidí partir el texto **por artículo completo**, usando el propio
número de artículo como marca de corte. Así cada fragmento queda con su
texto y su nota de vigencia juntos, sin importar cuánto mida.

En el camino encontré un problema que no había previsto: un artículo (el 134) aparecía mencionado dos veces en el texto, en dos lugares distintos, y
mi programa lo interpretaba como si fueran dos artículos separados con el
mismo número, lo que después generaba un error porque no podía haber dos
identificadores iguales en la base de datos. Tuve que agregar una
numeración correlativa para que cada fragmento tuviera un identificador
único aunque el número de artículo se repitiera. Fue un buen recordatorio
de que un texto legal real tiene irregularidades que uno no anticipa hasta
que las ve fallar.

### Cargar la base vectorial

Una vez que tuve los artículos bien separados, generé los embeddings con
Cohere y los guardé en ChromaDB de forma persistente (o sea, que queda
guardado en disco y no se pierde si reinicio el servidor). Esto lo corrí
como un script aparte, una sola vez, separado de la API: la API solo
consulta lo que ya está cargado, no vuelve a procesar el texto cada vez que
alguien hace una pregunta.

Terminé con 317 fragmentos de los 278 artículos de la ley (más los que
tienen sufijo "bis" o "ter", como el 92 bis del período de prueba).

---

## Construir la API

Acá pude reutilizar bastante de lo que había aprendido en el challenge
anterior del bootcamp: la estructura de FastAPI, el manejo de errores, el
guardrail de lenguaje inapropiado, la forma de loguear cada paso de una
consulta sin exponer datos sensibles. No arranqué de cero en eso.

Lo que sí fue nuevo para este proyecto:

**Que la misma pregunta genere siempre la misma respuesta.** Al principio
pensé que alcanzaba con bajar la "temperatura" del modelo a 0 (un parámetro
que controla cuánta variación tiene la respuesta). Pero aprendí que incluso
en 0 los modelos de lenguaje pueden variar un poco entre una llamada y
otra. La forma de garantizarlo de verdad fue guardar la respuesta la
primera vez que se genera una pregunta, y devolver esa misma respuesta
guardada si alguien vuelve a preguntar lo mismo (aunque lo escriba distinto,
con o sin tildes, con mayúsculas o no).

**Que responda siempre en español**, sin importar en qué idioma pregunten.
Esto lo agregué como instrucción directa en el prompt y lo probé
preguntando en inglés para confirmar que funcionaba.

**Que no use emojis.** Se lo pedí al modelo por instrucción, pero además
agregué un filtro que los elimina de la respuesta por si el modelo no
obedece. Aprendí que una instrucción en el prompt es una sugerencia, no una
garantía; un filtro después sí lo es.

**El aviso legal.** Como el tema es normativa laboral, me pareció importante
que cada respuesta aclarara que es información general y no un consejo
legal para un caso puntual.

---

## Calibrar cuándo el sistema debe responder

Esta fue, para mí, la parte más interesante de todo el proyecto, porque no
es algo que se resuelva "a ojo": hay que medir.

El sistema busca los artículos más parecidos a la pregunta y les asigna un
número entre 0 y 1 (cuanto más alto, más parecido). Pero necesitaba definir
a partir de qué número considero que la pregunta realmente tiene que ver
con la ley, y por debajo de cuál no.

Hice varias preguntas de prueba y anoté los números que me devolvía:

- Preguntas que sí eran sobre la ley (vacaciones, período de prueba,
  salario mínimo) dieron números entre 0.56 y 0.72.
- Preguntas que no tenían nada que ver (la capital de Francia, una receta
  de tarta) dieron entre 0.33 y 0.34.

Con esos datos elegí 0.45 como el punto de corte, porque queda justo en el
medio, lejos de los dos grupos. Antes de medir esto, había puesto un número
al azar (0.30) y con ese número una pregunta sobre una tarta de manzana
casi pasaba el filtro. Aprendí que un umbral sin medir es una apuesta, no
una decisión.

### Agregar reranking

Después de tener esto funcionando, me di cuenta —mirando los artículos que
el sistema citaba— de que a veces traía un artículo que no tenía mucho que
ver con la pregunta, solo porque compartía alguna palabra. Por ejemplo,
para "período de prueba" (que es cuando alguien recién empieza a trabajar)
el sistema a veces traía un artículo que habla de la "prueba" del contrato
en otro sentido, el de acreditar que existió.

Investigué y encontré que Cohere (el mismo proveedor de IA que ya estaba
usando) tiene un servicio aparte llamado "reranking", que agarra un grupo
de candidatos y los reordena comparando la pregunta contra cada uno de
forma más profunda que la búsqueda inicial. Lo probé con un script aparte
antes de meterlo en el sistema principal, para no arriesgar lo que ya
funcionaba, y medí el antes y el después:

- En la pregunta sobre período de prueba, el artículo confuso (el 50) bajó
  del segundo al tercer lugar, y apareció uno que tiene más sentido (el
  231, sobre preaviso, al que el propio artículo del período de prueba
  hace referencia).
- En la pregunta sobre vacaciones, salió un artículo genérico sobre
  antigüedad que no aportaba nada, y entró otro artículo del mismo tema de
  vacaciones.

Con esos números pude decidir con evidencia que valía la pena agregarlo, en
vez de agregarlo "porque quedaba bien" en la presentación.

Tuve que aprender algo más en el camino: el número que da el reranking no
se puede comparar directamente con el número de la búsqueda inicial, son
dos escalas distintas. Por eso terminé con dos filtros separados, cada uno
con su propio punto de corte medido por separado.

Un dato curioso que también aprendí en el camino: cuando busqué el nombre
del modelo de reranking a usar, resultó que el que aparecía en la
documentación que encontré primero (`rerank-v3.5`) ya estaba dado de baja
por Cohere. Tuve que armar un pequeño script que probara varios nombres
hasta encontrar cuál funcionaba con mi clave de acceso. Me pasó algo
parecido con el modelo de generación de texto en el challenge anterior. Ahí
entendí que trabajar con servicios de terceros implica que las cosas
cambian con el tiempo, y por eso dejé los nombres de los modelos en un
solo lugar del código, para poder actualizarlos fácilmente si vuelve a
pasar.

---

## Un caso que me pareció importante para IA Responsable

Durante las pruebas le pregunté al sistema cuánto es el salario mínimo
vital y móvil. La búsqueda encontró los artículos correctos de la ley (los
que hablan del salario mínimo) con el número de similitud más alto de
todas mis pruebas. Pero la respuesta que generó el modelo fue que la ley
define el concepto, pero no fija ningún monto (eso lo establece otro
organismo, no esta ley). Y el sistema marcó esa respuesta como "no
fundamentada" a pesar de que la búsqueda había encontrado muy bien los
artículos.

Me pareció un buen ejemplo de que el sistema no solo mide si encontró
algo parecido, sino que además el modelo revisa si con eso alcanza para
responder. Si solo hubiera confiado en el número de similitud, el sistema
habría marcado esa respuesta como confiable cuando en realidad el dato
concreto que pedía la pregunta no estaba en el texto.

---

## Adaptar el proyecto a la consigna final

Yo había empezado el proyecto con anticipación, antes de que nos
compartieran la consigna real del challenge. Cuando la leí, me di cuenta
de que tenía buena parte hecha, pero que la API no tenía los endpoints que
se pedían, que faltaba una evaluación formal y que no tenía ningún punto de
control humano. Así que antes de agregar nada nuevo, hice una lista de lo
que pedía la consigna punto por punto y la comparé con lo que ya tenía.

Lo primero fue reorganizar la API. Antes tenía un solo endpoint
(`/consultar`) que hacía todo junto: buscaba, filtraba y generaba la
respuesta. La consigna pedía poder ver qué artículos encuentra el sistema
**sin** que el modelo genere nada, así que separé la búsqueda en una
función propia y armé tres endpoints: `/health` para saber si el servicio
está bien, `/retrieve` para ver solo lo que se recupera, y `/ask` para la
pregunta completa. Lo bueno es que `/retrieve` y `/ask` usan exactamente la
misma función de búsqueda, así que lo que veo en uno es lo mismo que usa el
otro.

También pasé las instrucciones del modelo a lo que se llama "system
prompt", que es un lugar separado de la pregunta del usuario, y las ordené
por secciones: qué rol tiene, cómo usar la evidencia, qué hacer cuando no
hay información, qué no puede hacer y cómo tiene que ser la respuesta.

Antes de tocar nada guardé todo en Git, para poder volver atrás si rompía
algo.

---

## Agregar un control humano (Human in the Loop)

La consigna pedía identificar al menos una situación donde una persona
tenga que revisar antes de que el sistema siga. La primera idea que surgió
fue mandar a revisión todas las consultas sobre despidos, indemnizaciones o
sanciones, porque son temas delicados. Pero lo pensé y no me convenció: si
alguien pregunta cuánto preaviso le corresponde y la ley lo dice
clarísimo, frenar esa respuesta no protege a nadie, solo la demora. Además,
esas son de las preguntas más comunes, y si todas van a revisión, el
revisor termina aprobando sin leer.

Entonces decidí que lo que define si hace falta una persona no es el tema
de la pregunta, sino **qué tan buena es la evidencia**. Quedaron dos
criterios:

- Cuando entre los artículos que usa la respuesta hay uno **derogado**. La
  ley tiene 16 artículos derogados, varios por la reforma de este año, y
  alguien tiene que confirmar que la respuesta no se apoya en uno de ellos.
- Cuando la **confianza es baja**, o sea, cuando el mejor artículo tiene un
  puntaje de reranking entre 0.30 y 0.50. En mis mediciones, las preguntas
  que sí eran sobre la ley daban 0.57 o más, y las que no tenían nada que
  ver daban 0.11 o menos. Lo del medio es una zona gris que ninguna
  medición respalda, y ahí prefiero que decida una persona.

Cuando pasa alguna de esas dos cosas, el sistema genera la respuesta
igual, pero no la muestra: queda "pendiente de aprobación" hasta que un
revisor la aprueba o la rechaza.

Para probarlo busqué preguntas reales que cayeran en cada caso, y una me
sirvió mucho para entender por qué esto hace falta. Pregunté "¿me pueden
pagar con tickets de comida?" y el sistema respondió que no, "según el
artículo 131". Suena convincente, pero cuando fui a leer el artículo 131,
habla de los descuentos que no se pueden hacer sobre el sueldo, no de cómo
se paga. Lo que dice que el sueldo se paga en dinero es el artículo 105. La
respuesta sonaba segura, citaba un artículo real, y estaba mal
fundamentada. Ese es justo el tipo de error que un revisor tiene que
atajar.

---

## Evaluar el sistema

Hasta acá yo había probado el sistema con preguntas sueltas, pero la
consigna pedía una evaluación de verdad. Armé un grupo de 15 preguntas y,
para cada una, anoté leyendo la ley qué artículo la responde y qué dato
tiene que aparecer en la respuesta. Traté de meter trampas: preguntas con
palabras que la ley no usa ("aguinaldo" en vez de "sueldo anual
complementario"), preguntas donde el tema es de la ley pero el dato no
está, una pregunta que no tiene nada que ver y los dos casos de revisión
humana.

Medí tres cosas por separado: si el sistema encontró el artículo correcto,
si hizo lo que tenía que hacer (responder, negarse, cortar o mandar a
revisión) y si la respuesta era correcta y estaba respaldada por el texto.
Para esto último usé otro modelo como "juez" que le pone puntaje a cada
respuesta, y además las leí yo una por una comparándolas con la ley.

Los resultados fueron buenos: el sistema hizo lo correcto en las 15
preguntas y, con el reranking, encontró el artículo correcto en todas las
que tenían uno. Sin el reranking hubiera fallado una. Pero lo más
interesante fue lo que encontré al leer las respuestas a mano: en 3 de
ellas el modelo agregó frases que no estaban en los artículos que le había
pasado. Por ejemplo, en la del preaviso explicó "para qué sirve" el
preaviso, algo que la ley no dice, y dio un dato que sale de otro artículo
que ni siquiera se había recuperado. Las respuestas eran correctas en lo
principal y las frases agregadas sonaban razonables, por eso es difícil
darse cuenta.

Y el juez no se dio cuenta: les puso el puntaje máximo en "fundamentada" a
las tres, y también a la de los tickets. Aprendí que si uso el mismo modelo
para generar y para evaluar, comparte las mismas cegueras. El número del
juez me hubiera hecho creer que el sistema no inventaba nada. Esto me
confirmó que el control humano tiene que ser una persona y no otro modelo.

Con ese hallazgo ajusté el prompt: agregué una regla para que no sume
explicaciones ni consecuencias que no estén escritas en los artículos,
aunque sean ciertas. Volví a correr la evaluación y esta vez ninguna de las
12 respuestas agregó nada que no estuviera en el texto. Pero no fue gratis:
el modelo se volvió más literal, y en la pregunta de las horas extra copió
el artículo casi palabra por palabra, hasta con un error de tipeo que tiene
la ley. Aprendí que ajustar un prompt es un equilibrio: lo que gané en
fidelidad lo perdí un poco en claridad.

---

## Cuando el modelo se rompe

Durante las pruebas me pasó algo que no esperaba: una vez, en vez de
responder sobre las horas extra, el modelo devolvió miles de "3" seguidos.
Y lo peor es que esa respuesta quedó guardada, así que si alguien volvía a
preguntar lo mismo, iba a recibir esa basura. No lo pude repetir a
propósito, pero volvió a pasar varias veces, incluso con la temperatura en
0.

Como no lo puedo evitar, lo que hice fue detectarlo: el sistema revisa si
la respuesta es una repetición sin sentido y, si lo es, vuelve a intentar
hasta tres veces. Si igual falla, devuelve un error en vez de guardar algo
roto. En la evaluación pasó en una pregunta: falló dos veces seguidas y al
tercer intento salió bien. Aprendí que la temperatura en 0 no garantiza que
el modelo se comporte siempre igual, y que hay que revisar lo que devuelve
antes de confiar en eso.

---

## Pruebas automáticas y el límite de la API

En el medio del trabajo me quedé sin llamadas a Cohere: la clave gratuita
tiene un límite de 1000 por mes, y entre las pruebas y la evaluación se
agotó. Tuve que crear otra cuenta. Eso me hizo pensar distinto las pruebas
automáticas: en lugar de que cada prueba llame a Cohere, armé una versión
"falsa" de Cohere y de la base de datos que responde lo que cada prueba
necesita. Así las 28 pruebas corren en menos de un segundo, sin gastar
ninguna llamada, y además puedo simular cosas que con el servicio real no
puedo provocar a propósito, como que Cohere se caiga o que el modelo
devuelva la basura de los "3".

Como QA, hice algo que siempre hago cuando todas las pruebas pasan de
entrada: rompí el código a propósito para confirmar que las pruebas se dan
cuenta. Cambié el umbral de revisión humana y fallaron 6 pruebas; desactivé
el caché y fallaron 2. Una prueba que nunca falla no me dice nada.

---

## Cómo trabajé

No escribí todo el código de una sola vez ni de memoria. Fui construyendo
por partes: primero conseguí y revisé los datos, después armé el sistema de
búsqueda, después la generación de respuestas, y en cada parte probé que
funcionara antes de seguir con la siguiente. Varias veces encontré un
problema recién al probarlo con datos reales, no leyendo el código nomás
(el caso del artículo 134 repetido es un buen ejemplo: si no llegaba a
correr la carga real, no me hubiera dado cuenta).

Trabajé con la guía de un asistente de IA para escribir el código y
entender cada decisión técnica, porque no vengo del área de desarrollo.
Pero en cada paso pedí que me explicaran qué hacía cada parte y por qué,
y verifiqué con pruebas propias (no solo confiando en que "andaba") antes
de dar cada etapa por cerrada. Como QA, esa parte de verificar antes de
avalar es la que más cómodo me resultó, y traté de aplicarla en todo el
proceso, no solo al final.

---

## Qué me falta entender mejor

Soy honesto: hay conceptos que todavía tengo que masticar más, como el
detalle matemático de cómo se calculan los embeddings o cómo funciona por
dentro el modelo de reranking. Lo que sí puedo explicar con seguridad es
**qué decisión tomé, por qué la tomé, y qué medí para confirmar que era la
correcta** — que es, en definitiva, la forma de pensar que ya tenía de mi
trabajo como QA, aplicada a un tipo de sistema nuevo para mí.
