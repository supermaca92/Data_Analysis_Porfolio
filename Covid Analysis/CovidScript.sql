-- Check that the tables have been imported properly

--SELECT *
--FROM dbo.CovidDeaths$
--ORDER BY 3, 4;

--SELECT *
--FROM dbo.CovidVaccinations$
--ORDER BY 3, 4;

-- Select the Data that we are going to be using

-- DEATH DATA

SELECT Location,
		date,
		total_cases,
		new_cases,
		total_deaths,
		population
FROM CovidDeaths$
WHERE continent IS NOT NULL
ORDER BY 1, 2;



-- Looking at total cases vs total deaths --> progression through time.
-- Shows the likelihood of dying if you contract COVID in your country.

CREATE VIEW CasesvsDeaths AS
SELECT location,
		date,
		total_cases,
		total_deaths,
		(total_deaths/total_cases)*100 AS Death_Percentage
FROM CovidDeaths$
WHERE continent IS NOT NULL
--ORDER BY 1, 2;

-- Lookint at total cases vs population
-- Shows the percentage of the population that has contracted covid

CREATE VIEW CovidCases AS
SELECT location,
		date,
		total_cases,
		population,
		(total_cases/population)*100 AS case_percentage
FROM CovidDeaths$
WHERE continent IS NOT NULL
--ORDER BY 1, 2;

-- What countries have the highest infection rates? (when compared to the population)

SELECT location,
		MAX(total_cases) AS highest_infection_count,
		Population,
		MAX((total_cases/population))*100 AS max_infection_percentage
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY location, population
ORDER BY max_infection_percentage DESC;

-- Showing countries with the highest death count per population

SELECT location,
		MAX(CAST(total_deaths AS INT)) AS highest_death_count
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY location
ORDER BY highest_death_count DESC;


-- LET'S BREAK THINGS DOWN BY CONTINENT

-- What continents have the highest infection rates? (when compared to the population)

SELECT location,
		MAX(total_cases) AS highest_infection_count,
		Population,
		MAX((total_cases/population))*100 AS max_infection_percentage
FROM CovidDeaths$
WHERE continent IS NULL
GROUP BY location, population
ORDER BY max_infection_percentage DESC;

-- Showing continents with the highest death count per population

SELECT continent,
		MAX(CAST(total_deaths AS INT)) AS TotalDeathCount
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY continent
ORDER BY TotalDeathCount DESC;

-- GLOBAL NUMBERS

-- By date

SELECT date,
		SUM(new_cases) AS TotalCases,
		SUM(CAST(new_deaths AS INT)) AS TotalDeaths,
		SUM(CAST(new_deaths AS INT))/SUM(new_cases) * 100 AS DeathPercentage
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY date
ORDER BY 1, 2;

-- Totals

SELECT SUM(new_cases) AS TotalCases,
		SUM(CAST(new_deaths AS INT)) AS TotalDeaths,
		SUM(CAST(new_deaths AS INT))/SUM(new_cases) * 100 AS DeathPercentage
FROM CovidDeaths$
WHERE continent IS NOT NULL
ORDER BY 1, 2;


-- VACCINATION AND TEST DATA

-- Total population vs vaccination

SELECT dea.continent,
		dea.location,
		dea.date,
		dea.population,
		vac.new_vaccinations,
		SUM(CAST(vac.new_vaccinations AS BIGINT)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS RollingVaccines,
		vac.people_fully_vaccinated
FROM CovidDeaths$ AS dea
JOIN CovidVaccinations$ AS vac
	ON dea.location = vac.location 
	AND dea.date = vac.date
WHERE dea.continent IS NOT NULL
ORDER BY 2,3;

-- Use CTE 

WITH PeopleVac (Continent, Location, Date, Population, New_vaccinations, RollingVaccinated, Fully_Vaccinated)
AS(
	SELECT dea.continent,
		dea.location,
		dea.date,
		dea.population,
		vac.new_vaccinations,
		SUM(CAST(vac.new_vaccinations AS INT)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS RollingVaccines,
		vac.people_fully_vaccinated
FROM CovidDeaths$ AS dea
JOIN CovidVaccinations$ AS vac
	ON dea.location = vac.location 
	AND dea.date = vac.date
WHERE dea.continent IS NOT NULL)
SELECT location,
	MAX(RollingVaccinated/Population)*100 AS PercentageVaccinated,
	MAX(Fully_Vaccinated/Population)*100 AS PercentageFullyVaccinated
FROM PeopleVac
GROUP BY location;


-- Create views for further processing in Tableau

CREATE View PercentPopulationVaccinated AS
WITH PeopleVac (Continent, Location, Date, Population, New_vaccinations, RollingVaccinated, Fully_Vaccinated)
AS(
	SELECT dea.continent,
		dea.location,
		dea.date,
		dea.population,
		vac.new_vaccinations,
		SUM(CAST(vac.new_vaccinations AS INT)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS RollingVaccines,
		vac.people_fully_vaccinated
FROM CovidDeaths$ AS dea
JOIN CovidVaccinations$ AS vac
	ON dea.location = vac.location 
	AND dea.date = vac.date
WHERE dea.continent IS NOT NULL)
SELECT location,
	MAX(RollingVaccinated/Population)*100 AS PercentageVaccinated,
	MAX(Fully_Vaccinated/Population)*100 AS PercentageFullyVaccinated
FROM PeopleVac
GROUP BY location;

CREATE VIEW CovidTotals AS
SELECT SUM(new_cases) AS TotalCases,
		SUM(CAST(new_deaths AS INT)) AS TotalDeaths,
		SUM(CAST(new_deaths AS INT))/SUM(new_cases) * 100 AS DeathPercentage
FROM CovidDeaths$
WHERE continent IS NOT NULL;

CREATE VIEW CountriesDeathCounts AS
SELECT location,
		MAX(CAST(total_deaths AS INT)) AS death_count
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY location
--ORDER BY highest_death_count DESC;

CREATE VIEW GlobalDailyTotals AS
SELECT date,
		SUM(new_cases) AS TotalCases,
		SUM(CAST(new_deaths AS INT)) AS TotalDeaths,
		SUM(CAST(new_deaths AS INT))/SUM(new_cases) * 100 AS DeathPercentage
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY date


CREATE VIEW InfectionRatesContinents AS
SELECT location,
		MAX(total_cases) AS highest_infection_count,
		Population,
		MAX((total_cases/population))*100 AS max_infection_percentage
FROM CovidDeaths$
WHERE continent IS NULL
GROUP BY location, population;

CREATE VIEW DeathRatesContinents AS
SELECT continent,
		MAX(CAST(total_deaths AS INT)) AS TotalDeathCount
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY continent;

CREATE VIEW CountriesInfectionRates AS
SELECT location,
		MAX(total_cases) AS highest_infection_count,
		Population,
		MAX((total_cases/population))*100 AS max_infection_percentage
FROM CovidDeaths$
WHERE continent IS NOT NULL
GROUP BY location, population
