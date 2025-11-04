package bibliometria;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class SorensenDiceCoefficient {

    /**
     * Calcula el coeficiente de Sørensen-Dice entre dos textos.
     * El coeficiente es una medida de similitud entre dos conjuntos y se calcula como:
     * 2 * |X ∩ Y| / (|X| + |Y|)
     * donde X e Y son los conjuntos de palabras de los dos textos.
     *
     * @param text1 El primer texto.
     * @param text2 El segundo texto.
     * @return El coeficiente de Sørensen-Dice (entre 0.0 y 1.0).
     */
    public static double calculate(String text1, String text2) {
        if (text1 == null || text2 == null) {
            throw new IllegalArgumentException("Los textos no pueden ser nulos");
        }

        Set<String> set1 = Arrays.stream(text1.toLowerCase().split("\\s+"))
                                 .collect(Collectors.toSet());
        Set<String> set2 = Arrays.stream(text2.toLowerCase().split("\\s+"))
                                 .collect(Collectors.toSet());

        if (set1.isEmpty() && set2.isEmpty()) {
            return 1.0; // Dos textos vacíos son idénticos
        }

        Set<String> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);

        return (2.0 * intersection.size()) / (set1.size() + set2.size());
    }
}
