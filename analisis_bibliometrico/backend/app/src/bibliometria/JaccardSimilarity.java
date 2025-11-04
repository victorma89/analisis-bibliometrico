package bibliometria;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class JaccardSimilarity {

    /**
     * Calcula la similitud de Jaccard entre dos textos.
     * La similitud de Jaccard se define como el tamaño de la intersección de los conjuntos de palabras
     * dividido por el tamaño de la unión de los conjuntos de palabras.
     *
     * @param text1 El primer texto.
     * @param text2 El segundo texto.
     * @return El coeficiente de similitud de Jaccard (entre 0.0 y 1.0).
     */
    public static double calculate(String text1, String text2) {
        if (text1 == null || text2 == null) {
            throw new IllegalArgumentException("Los textos no pueden ser nulos");
        }

        // Tokenización simple: dividir por espacios y convertir a minúsculas
        Set<String> set1 = Arrays.stream(text1.toLowerCase().split("\\s+"))
                                 .collect(Collectors.toSet());
        Set<String> set2 = Arrays.stream(text2.toLowerCase().split("\\s+"))
                                 .collect(Collectors.toSet());

        if (set1.isEmpty() && set2.isEmpty()) {
            return 1.0; // Dos textos vacíos son idénticos
        }

        Set<String> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);

        Set<String> union = new HashSet<>(set1);
        union.addAll(set2);

        if (union.isEmpty()) {
            return 0.0; // Evitar división por cero si ambos textos solo contienen espacios
        }

        return (double) intersection.size() / union.size();
    }
}
