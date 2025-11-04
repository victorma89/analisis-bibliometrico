package bibliometria;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CosineSimilarity {

    /**
     * Calcula la similitud del coseno entre dos textos utilizando ponderación TF-IDF.
     *
     * @param text1 El primer texto.
     * @param text2 El segundo texto.
     * @param documents Una lista de todos los documentos (textos) en el corpus para calcular el IDF.
     * @return La similitud del coseno (entre 0.0 y 1.0).
     */
    public static double calculate(String text1, String text2, List<String> documents) {
        if (text1 == null || text2 == null || documents == null) {
            throw new IllegalArgumentException("Los textos y el corpus no pueden ser nulos");
        }

        // Crear el vocabulario completo a partir de ambos textos
        Set<String> vocabulary = Stream.concat(
                Arrays.stream(text1.toLowerCase().split("\\s+")),
                Arrays.stream(text2.toLowerCase().split("\\s+"))
        ).collect(Collectors.toSet());

        // Calcular los vectores TF-IDF para cada texto
        Map<String, Double> tfIdfVector1 = getTfIdfVector(text1, documents, vocabulary);
        Map<String, Double> tfIdfVector2 = getTfIdfVector(text2, documents, vocabulary);

        // Calcular el producto punto
        double dotProduct = 0.0;
        for (String term : vocabulary) {
            dotProduct += tfIdfVector1.getOrDefault(term, 0.0) * tfIdfVector2.getOrDefault(term, 0.0);
        }

        // Calcular las magnitudes de los vectores
        double magnitude1 = 0.0;
        for (double value : tfIdfVector1.values()) {
            magnitude1 += Math.pow(value, 2);
        }
        magnitude1 = Math.sqrt(magnitude1);

        double magnitude2 = 0.0;
        for (double value : tfIdfVector2.values()) {
            magnitude2 += Math.pow(value, 2);
        }
        magnitude2 = Math.sqrt(magnitude2);

        // Evitar división por cero
        if (magnitude1 == 0.0 || magnitude2 == 0.0) {
            return 0.0;
        }

        return dotProduct / (magnitude1 * magnitude2);
    }

    /**
     * Calcula el vector TF-IDF para un texto dado.
     */
    private static Map<String, Double> getTfIdfVector(String text, List<String> documents, Set<String> vocabulary) {
        List<String> terms = Arrays.asList(text.toLowerCase().split("\\s+"));
        Map<String, Double> tfIdfVector = new HashMap<>();

        for (String term : vocabulary) {
            double tf = calculateTf(terms, term);
            double idf = calculateIdf(documents, term);
            tfIdfVector.put(term, tf * idf);
        }
        return tfIdfVector;
    }

    /**
     * Calcula la Frecuencia de Término (TF) para un término en un documento.
     */
    private static double calculateTf(List<String> terms, String term) {
        long count = terms.stream().filter(t -> t.equals(term)).count();
        return (double) count / terms.size();
    }

    /**
     * Calcula la Frecuencia Inversa de Documento (IDF) para un término en el corpus.
     */
    private static double calculateIdf(List<String> documents, String term) {
        long docsContainingTerm = documents.stream()
                .filter(doc -> Arrays.asList(doc.toLowerCase().split("\\s+")).contains(term))
                .count();
        // Se suma 1 al denominador para evitar división por cero si el término no está en ningún documento
        return Math.log((double) documents.size() / (docsContainingTerm + 1));
    }
}
