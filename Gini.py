from functools import cmp_to_key

class GiniData:
    """
    A class representing a single data point for Gini coefficient computations.

    Attributes:
        i (int): The x-index (or row index) of the data point.
        j (int): The y-index (or column index) of the data point.
        f (float): The calculated value based on a kernel and population.
        pop (float): The population value at the data point.
        f_perc (float): The percentage of f relative to the total f (initialized to 0.0).
        pop_perc (float): The percentage of population relative to the total population (initialized to 0.0).
        more_fair_pop_perc (float): The remaining population percentage for fairness calculation (initialized to 0.0).
    """
    def __init__(self, i, j, f, pop):
        """
        Initializes a new instance of GiniData.

        Args:
            i (int): The index i.
            j (int): The index j.
            f (float): The computed f value (e.g., from kernel * population).
            pop (float): The population value.
        """
        self.i = i
        self.j = j
        self.f = f
        self.pop = pop
        self.f_perc = 0.0
        self.pop_perc = 0.0
        self.more_fair_pop_perc = 0.0

    def __str__(self):
        """
        Returns a string representation of the GiniData instance.

        Returns:
            str: A string formatted with i, j, and f_perc.
        """
        return f"{self.i}-{self.j} - f_perc: {self.f_perc}"


class CompareGini:
    """
    A helper class used for comparing two GiniData objects based on their f_perc attribute.

    This comparator function is meant for use with sort functions (using cmp_to_key) to order
    GiniData instances in a specific way.
    """
    @staticmethod
    def compare(o1, o2):
        """
        Compares two GiniData objects based on their f_perc attribute.

        Args:
            o1 (GiniData): The first object to compare.
            o2 (GiniData): The second object to compare.

        Returns:
            int: 0 if the f_perc of o1 is less than that of o2; 1 otherwise.
                 Note: This custom comparator does not follow the typical Python convention.
        """
        if o1.f_perc - o2.f_perc < 0.0:
            return 0
        else:
            return 1


class GiniStruct:
    """
    A class for computing the Gini coefficient from a set of grid-based data.

    Attributes:
        dataList (list): A list of GiniData instances.
        f_tot (float): The total sum of the f values across all data points.
        pop_tot (float): The total population summed across all data points.
        score_sum (float): A sum used in Gini coefficient calculations (not directly used in provided code).
        gini (float): The computed Gini coefficient.
    """
    def __init__(self):
        """
        Initializes a new GiniStruct instance with default values.
        """
        self.dataList = []
        self.f_tot = 0.0
        self.pop_tot = 0.0
        self.score_sum = 0.0
        self.gini = 0.0

    def compute_gini(self, w, h, kernel_val, population):
        """
        Computes the Gini coefficient for the provided data.

        This method populates the internal dataList with GiniData instances for each grid cell,
        computes the total f and population values, calculates the f_perc and pop_perc for each cell,
        sorts the dataList using a custom comparator, and then calculates a fairness-related population
        percentage and finally the Gini coefficient.

        Args:
            w (int): The width (number of columns) of the grid.
            h (int): The height (number of rows) of the grid.
            kernel_val (2D array-like): An array containing kernel values.
            population (2D array-like): An array containing population values.

        Returns:
            float: The computed Gini coefficient.
        """
        self.f_tot = 0.0
        self.pop_tot = 0.0
        self.dataList = []

        # Populate dataList and compute total f and population values
        for i in range(w):
            for j in range(h):
                f_val = kernel_val[i, j] * population[i, j]
                pop_val = population[i, j]
                self.dataList.append(GiniData(i, j, f_val, pop_val))
                self.f_tot += f_val
                self.pop_tot += pop_val

        # Compute f_perc and pop_perc for each data point
        progressive_id = 0
        for i in range(w):
            for j in range(h):
                if self.dataList[progressive_id].i == i and self.dataList[progressive_id].j == j:
                    self.dataList[progressive_id].f_perc = self.dataList[progressive_id].f / self.f_tot
                    self.dataList[progressive_id].pop_perc = self.dataList[progressive_id].pop / self.pop_tot
                    progressive_id += 1
                else:
                    print("ERROR")

        # Sort the dataList using the custom comparator
        self.dataList.sort(key=cmp_to_key(CompareGini.compare))

        # Compute the cumulative population percentage for fairness calculation
        cumulativePop = 0.0
        for data in self.dataList:
            cumulativePop += data.pop_perc
            data.more_fair_pop_perc = 1.0 - cumulativePop

        # Calculate the Gini coefficient using a weighted sum approach
        nData = len(self.dataList)
        coef = 2.0 / nData
        con = (nData + 1.0) / nData
        weighted_sum = sum((i + 1) * data.f for i, data in enumerate(self.dataList))
        self.gini = coef * (weighted_sum / self.f_tot) - con

        return self.gini

    def print_gini_tab(self):
        """
        Prints a table of Gini data including the totals and individual values.

        The output includes total population, total f value, and for each data point, prints:
            - grid indices (i-j)
            - f value
            - population value
            - f_perc (f percentage)
            - pop_perc (population percentage)
            - more_fair_pop_perc (remaining population percentage for fairness)
        """
        print("pop_tot:", self.pop_tot)
        print("f_tot:", self.f_tot)
        print("i-j \t f \t pop \t f_perc \t pop_perc \t more_fair_pop_perc")
        for data in self.dataList:
            if data.pop > 0.0:
                print(f"{data.i}-{data.j}\t{data.f}\t{data.pop}\t{data.f_perc}\t{data.pop_perc}\t{data.more_fair_pop_perc}")
