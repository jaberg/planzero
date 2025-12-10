Read Me – Additional information on GHG CSV files


IPCC categories csv file

CategoryID column: Sort that column from smallest to largest to have the same category order than what is 
presented in other products (e.g. National GHG Inventory Report: Tables A9-2, A9-3 & A11-2 to A11-28, Excel 
tables “Tables-IPCC-Sector” in ECCC Data Mart).

Use CategoryID = 0 to get the Total for Canada or for a Province or Territory, for a given year.

Total column: "y" means that the source of Category is an aggregated total. No value in that column means 
that it is the lowest level of breakdown. The total of the emissions can be calculated by adding all categories 
that do not have a value in the Total column.

“x” value in gases columns: Indicates that data has been suppressed to respect confidentiality.



Economic sector csv file

Index column: Sort that column from smallest to largest to have the same category order as what is presented 
in other products (e.g. National GHG Inventory Report: Tables A10-2 & A12-2 to A12-15, 
Excel tables “Tables-Canadian-Economic-Sector” in ECCC Data Mart).

Use Index = 0 to get the National, Provincial or Territorial Inventory Total for a given year.

Total column: a “y” (for “yes”) indicated that the source, category or sub-category is an aggregated total. 
No value in that column means that it is the lowest level of breakdown.

“x” value in gases columns: Indicates that data has been suppressed to respect confidentiality.


