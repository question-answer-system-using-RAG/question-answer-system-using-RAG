# Эксперименты с retrieval
Были выбраны три базовые модели: SentenceTransformer('intfloat/multilingual-e5-large'), SentenceTransformer('BAAI/bge-large-en-v1.5'), BM25 и четыре реранкера: cross-encoder/ms-marco-MiniLM-L-12-v2, BAAI/bge-reranker-base, nboost/pt-tinybert-msmarco, sentence-transformers/msmarco-distilbert-base-tas-b. Также аналитически было выделено количество фрагментов, которые целесообразно вытягивать из массивов документации: 4 и 6 фрагментов на тестирование. Обоснование этого лежит в ноутбуке eda.

Итого, было проведено 30 экспериментов с retrieval частью: с каждой из три базовых моделей 1 вариант без реранкера и 4 варианта с различным реранкерами, а также с вытягиванием 4 и 6 фрагментов для каждого из вариантов. Оцнка релевантности фрагментов проведены сторонней llm по пятибальной шкале, на данном этапе разработки перевела в бинарную шкалу (релевантен/не релевантен).

Оценку проводим в срелующих метриках: Recall@4/6, Precision@4/6, MRR@4/6, DCG@4/6. Результат сведен в таблицу.

Базовая модель	Реранкер	Recall@4	Recall@6	Precision@4	Precision@6	MRR@4	MRR@6	DCG@4	DCG@6
SentenceTransformer('intfloat/multilingual-e5-large')	-	0.84	1	0.41	0.34	0.80	0.81	0.96	0.95
	cross-encoder/ms-marco-MiniLM-L-12-v2	1	1	0.46	0.37	0.83	0.84	0.97	0.96
	BAAI/bge-reranker-base	1	1	0.46	0.38	0.86	0.86	0.97	0.96
	nboost/pt-tinybert-msmarco	1	1	0.37	0.31	0.68	0.69	0.95	0.94
	sentence-transformers/msmarco-distilbert-base-tas-b	1	1	0.21	0.20	0.31	0.33	0.92	0.91
SentenceTransformer('BAAI/bge-large-en-v1.5')	-	0.79	1	0.59	0.53	0.85	0.86	0.97	0.95
	cross-encoder/ms-marco-MiniLM-L-12-v2	0.87	1	0.44	0.37	0.78	0.78	0.97	0.96
	BAAI/bge-reranker-base	0.87	1	0.46	0.38	0.80	0.80	0.97	0.96
	nboost/pt-tinybert-msmarco	0.83	1	0.36	0.30	0.63	0.64	0.95	0.93
	sentence-transformers/msmarco-distilbert-base-tas-b	0.75	1	0.18	0.19	0.32	0.35	0.94	0.90
BM25	-	0.81	1	0.48	0.42	0.77	0.77	0.97	0.94
	cross-encoder/ms-marco-MiniLM-L-12-v2	1	1	0.45	0.39	0.76	0.76	0.81	0.83
	BAAI/bge-reranker-base	1	1	0.48	0.42	0.78	0.77	0.85	0.84
	nboost/pt-tinybert-msmarco	1	1	0.48	0.42	0.77	0.77	0.86	0.87
	sentence-transformers/msmarco-distilbert-base-tas-b	1	1	0.20	0.21	0.35	0.41	0.46	0.53
