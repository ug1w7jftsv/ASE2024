# Data_ASE2024

#### Dataset structure：

| -- arthas

| ------ Multi_trans/  # Multiple translation (NLLB、M2M100、Google、Tencent)

| ---------- issues.json  # Multiple translated issues

| ---------- commits.json  # Multiple translated commits

| ------ NLLB_trans/ 

| ------ M2M100_trans/

| ------ Google_trans/

| ------ Tencent_trans/

| ------ issues.json  # Untranslated issues

| ------ commits.json  # Untranslated commits

| ------ links.json  # Golden links

| -- (...other 16 projects)



#### Structure of Multiple translated artifacts: 

- An example from `arthas/Multi_trans/issues.json` are as follows.  

```json
"1": {
    // The sentences in summary and description are from Tencent translate (the optimal translator we chose).
    "summary": "Jad command adds options to support the display of line numbers",
    "description": "",
    "summary_bilingual_indexes": [ // Index of bilingual sentences in summary
        0
    ],
    "description_bilingual_indexes": [], // Index of bilingual sentences in description
    "summary_multi_trans": [
        [
            "jad command adds option support to display row number", // NLLB translation
            "Jad Command Add Options Support Display Line Number", // M2M100 translation
            "The jad command adds options to support displaying line numbers.", // Google translation
            "Jad command adds options to support the display of line numbers" // Tencent translation
        ]
    ],
    "description_multi_trans": [],
    "biterm": [
        "commandjad linenumber", // Biterms from summary
        "" // Biterms from description
    ]
}
```

