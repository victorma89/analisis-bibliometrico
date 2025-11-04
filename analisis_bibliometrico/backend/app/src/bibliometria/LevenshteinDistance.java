package bibliometria;

public class LevenshteinDistance {

    /**
     * Calcula la distancia de Levenshtein entre dos cadenas de texto.
     * La distancia de Levenshtein es el número mínimo de ediciones de un solo carácter
     * (inserciones, eliminaciones o sustituciones) necesarias para cambiar una palabra por otra.
     *
     * @param s1 La primera cadena.
     * @param s2 La segunda cadena.
     * @return La distancia de Levenshtein entre las dos cadenas.
     */
    public static int calculate(String s1, String s2) {
        if (s1 == null || s2 == null) {
            throw new IllegalArgumentException("Las cadenas no pueden ser nulas");
        }

        int[][] dp = new int[s1.length() + 1][s2.length() + 1];

        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i == 0) {
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;
                } else {
                    int cost = (s1.charAt(i - 1) == s2.charAt(j - 1)) ? 0 : 1;
                    dp[i][j] = Math.min(dp[i - 1][j] + 1,          // Eliminación
                                      Math.min(dp[i][j - 1] + 1,  // Inserción
                                               dp[i - 1][j - 1] + cost)); // Sustitución
                }
            }
        }
        return dp[s1.length()][s2.length()];
    }

    /**
     * Normaliza la distancia de Levenshtein a un valor entre 0.0 y 1.0,
     * donde 1.0 significa que las cadenas son idénticas.
     *
     * @param s1 La primera cadena.
     * @param s2 La segunda cadena.
     * @return El valor de similitud normalizado.
     */
    public static double normalizedSimilarity(String s1, String s2) {
        int maxLength = Math.max(s1.length(), s2.length());
        if (maxLength == 0) {
            return 1.0;
        }
        return 1.0 - (double) calculate(s1, s2) / maxLength;
    }
}
