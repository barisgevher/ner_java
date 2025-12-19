package com.example.ner.runner;


import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;

import java.nio.file.Path;
import java.nio.file.Paths;

public class ManualNerTest {
    public static void main(String[] args) throws Exception{

        System.out.println("=== DJL Turkish NER Manual Test ===");

        // Model dizin (LOCAL HuggingFace model)
        Path modelDir = Paths.get(
                "C:/turkish-ner-system/models"
        );

        //Criteria: model nasıl yüklenecek
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("C:/turkish-ner-system/inference-java/models"))
                        .optEngine("PyTorch")
                        .optTranslator(new NoopTranslator()) // sadece forward test için
                        .build();






        // modeli yükle
        try(ZooModel<NDList, NDList> model = criteria.loadModel()){
            System.out.println("Model yüklendi: " + model.getName());

            // predictor oluştur
            try(Predictor<NDList, NDList> predictor = model.newPredictor()){

                NDManager manager = NDManager.newBaseManager();

                //dummy input(sadece forward testi
                NDArray inputIds = manager.create(new long[][]{{101, 102}});
                NDArray attentionMask = manager.create(new long[][]{{1, 1}});
                NDArray tokenTypeIds = manager.create(new long[][]{{0, 0}});

                NDList input = new NDList(inputIds, attentionMask, tokenTypeIds);


                //forward pass

                NDList output = predictor.predict(input);

                System.out.println("Forward pass başarılı");
                System.out.println("Output tensor sayısı: " + output.size());
                System.out.println("ilk tensor shapei: " + output.get(0).getShape());
            }
        }
        System.out.println("=== TEST BAŞARIYLA TAMAMLANDI ===");
    }
}
