import json
import random
import logging

from config import env
import showdown.battle_bots.PokeBattle.utility as utility
from itertools import combinations

logger = logging.getLogger(__name__)

class Chromosome:
    mutation_rate = float(env("MUTATION_RATE"))

    def __init__(self, value: float, variance: float):
        self.value = value
        self.variance = variance

    @staticmethod
    def random_mutation(chromosome: 'Chromosome') -> 'Chromosome':
        "Applies random mutation proportional to the value"
        if random.random() <= Chromosome.mutation_rate:
            return chromosome

        # the value of the mutation is a random percentage of the variance, whose sign is randomly chosen
        mutation = (random.random() * random.choice([-1, 1])) * chromosome.variance
        variance = utility.avg([mutation, chromosome.variance])
        return Chromosome(chromosome.value * mutation / 100, variance)



class Genome:
    def __init__(self, genes: dict[str, Chromosome], parent_score: float = 0):
        self.genes = genes
        self.parent_score = parent_score
        self.score: float
    
    @classmethod
    def from_file(cls, filename: str | None = None) -> 'Genome':
        if not filename:
            filename = "base_genome"
        file = open(f"data/evolution/{filename}.json", "r")
        data: dict[str, dict[str, float] | str] = json.load(file)
        genes: dict[str, Chromosome] = {}
        for element in data:
            if element == "parent_score": continue
            genes[element] = Chromosome(float(data[element]["value"]), float(data[element]["variance"]))

        return Genome(genes, float(data.get("parent_score", 0)))

    def value(self, gene: str) -> float | int:
        "returns the value of the gene"
        if not gene:
            return 0
        return self.genes[gene].value

    def __getitem__(self, item_name: str) -> float | int:
        return self.value(item_name)
    
    def variance(self, gene: str) -> float | int:
        "returns the variance/instability of the gene"
        return self.genes[gene].variance

    def save(self, generation: int, genome_index: int):
        data = dict()
        stable = "Stable_"

        for gene in self.genes:
            gene_map = {"value": self.value(gene), "variance": self.variance(gene)}
            data[gene] = gene_map
            
            # doesn't print stable if the variance isn't
            if self.variance != 0:
                stable = ""
        file = open(f"data/evolution/{stable}gen{generation}_n{genome_index}.json", "w")
        json.dump(data, file, indent=4)
            

class Evolution:
    def __init__(self, genomes: list[Genome] | None = None, generation: int = 0):
        if genomes:
            self.genomes = genomes
            self.population_size: int = len(genomes)
        else:
            size = env("POPULATION_SIZE")
            assert isinstance(size, str)
            self.population_size = int(size)
            self.genomes = Evolution.create_genomes(self.population_size)
        self.generation = generation
        logger.debug(f"Generation {self.generation} created, population size: {self.population_size}")

    @staticmethod
    def create_genomes(n: int, base: Genome = Genome.from_file()) -> list[Genome]:
        "Creates n chromosomes from a base Chromosome"
        logger.debug(f"Created {n} new genomes with a base of score {base.parent_score or 0}")
        genomes = []
        for _ in range(0, n):
            gen: dict[str, Chromosome] = {}
            for gene in base.genes:
                gene_value = base.value(gene) + random.gauss(0, pow(base.variance(gene), 2))
                gen[gene] = Chromosome(gene_value, base.variance(gene))
            genomes.append(Genome(gen))
        return genomes

    @staticmethod
    def recombine(genome1: Genome, genome2: Genome) -> Genome:
        "Recombination procedure for two genomes (crossover)"
        genes = {}
        for gene in genome1.genes:
            gene_value = (genome1.value(gene) + genome2.value(gene)) / 2

            max_gene = max(genome1, genome2, key=lambda x: x.value(gene)).value(gene)
            min_gene = min(genome1, genome2, key=lambda x: x.value(gene)).value(gene)
            value_variance = utility.scale_range(gene_value, [min_gene, max_gene], [0, 1]) / 100
            gene_variance = value_variance * (genome1.variance(gene) * genome2.variance(gene) / 100)

            genes[gene] = Chromosome.random_mutation(Chromosome(gene_value, gene_variance))

        parent_score = utility.avg([genome1.parent_score, genome2.parent_score])
        return Genome(genes, parent_score)

    def culling(self) -> list[Genome]:
        """
        Next generation makeup tecnique, all the genomes that scored lower than their parent average are discarded
        """
        genomes = [genome for genome in self.genomes if genome.score >= genome.parent_score]
        return genomes

    def selection(self, genomes: list[Genome] | None = None) -> list[tuple[Genome, Genome]]:
        if not genomes:
            genomes = self.genomes
        return list(combinations(genomes, 2))

    def next_generation(self) -> 'Evolution':
        "Creates the next generation from the current one"
        logger.debug(f"Generating generation {self.generation+1}")
        genomes = self.culling() #performs culling
        couples = self.selection(genomes) #selects the couples that will create offsprings
        logger.debug(f"Culling procedure let {len(genomes)} survive the selection, {len(couples)} offsprings will be generated")

        # offsprings generation
        offsprings: list[Genome] = []
        for couple in couples:
            offspring = self.recombine(couple[0], couple[1])
            offsprings.append(offspring)
        
        # Adds offsprings to prevent population collapse
        if len(offsprings) < self.population_size:
            best_genome = max(genomes, key=lambda genome: genome.score)

            # adds offsprings as varaitions of the best scoring genome
            offsprings += Evolution.create_genomes(self.population_size - len(offsprings), best_genome)

        return Evolution(offsprings, self.generation + 1)

    

