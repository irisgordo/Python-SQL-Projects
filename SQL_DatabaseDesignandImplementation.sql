# customer 
CREATE TABLE customer (
	CustomerID 			NUMERIC(4)		PRIMARY KEY,
    CustomerName		VARCHAR(25)		NOT NULL,
    CustomerAddress 	VARCHAR(30)		NOT NULL,
    CustomerCity 		VARCHAR(20)		NOT NULL,
    CustomerState		CHAR(2)			NOT NULL,
    CustomerPostalCode 	VARCHAR(10)	
);

# territory
CREATE TABLE territory (
	TerritoryID			NUMERIC(3,1)	PRIMARY KEY,
    TerritoryName		VARCHAR(50)		NOT NULL
);

# doesbusinessin
CREATE TABLE doesbusinessin (
	CustomerID			NUMERIC(4)		PRIMARY KEY,
    TerritoryID			NUMERIC(4)		PRIMARY KEY,
    FOREIGN KEY (CustomerID) REFERENCES customer(CustomerID),
    FOREIGN KEY (TerritoryID) REFERENCES territory(TerritoryID)
);

# customershipaddress
CREATE TABLE customershipaddress (
	ShipAddressID		NUMERIC(4)		PRIMARY KEY,
    CustomerID			NUMERIC(4)		NOT NULL,
    TerritoryID			NUMERIC(4)		NOT NULL,
    ShipAddress			VARCHAR(30)		NOT NULL,
    ShipCity			VARCHAR(20)		NOT NULL,
    ShipState			CHAR(2)			NOT NULL,
    ShipZip				VARCHAR(10)		,
    ShipDirections		VARCHAR(50)		,
    FOREIGN KEY (CustomerID) REFERENCES customer(CustomerID),
    FOREIGN KEY (TerritoryID) REFERENCES territory(TerritoryID)
);

# salesperson
CREATE TABLE salesperson (
	SalespersonID			NUMERIC(4)		PRIMARY KEY,
    SalespersonName			VARCHAR(20)		NOT NULL,
    SalespersonTelephone 	CHAR(10)		,
    SalespersonFax			CHAR(10)		,
    SalespersonAddress		VARCHAR(30)		,
    SalespersonCity			VARCHAR(20)		,
    SalespersonState		CHAR(2)			,
    SalespersonZip			VARCHAR(10)		,
    SalesTerritoryID		NUMERIC(4)		NOT NULL,
	FOREIGN KEY (SalesTerritoryID) REFERENCES territory(TerritoryID)
);


# invoice
CREATE TABLE invoice (
	InvoiceID			NUMERIC(4)		PRIMARY KEY,
    CustomerID			NUMERIC(4)		NOT NULL,
    InvoiceDate			DATE			NOT NULL,
    FulfillmentDate		DATE			NOT NULL,
    SalespersonID		NUMERIC(4)		NOT NULL,
    ShipAdrsID			NUMERIC(4)		NOT NULL,
    FOREIGN KEY (CustomerID) REFERENCES customer(CustomerID),
    FOREIGN KEY (SalespersonID) REFERENCES salesperson(SalespersonID),
    FOREIGN KEY (ShipAdrsID) REFERENCES customershipaddress(ShipAddressID)
);

# payment
CREATE TABLE payment (
	PaymentID		NUMERIC(4)			PRIMARY KEY,
    InvoiceID		NUMERIC(4)			NOT NULL,
    PaymentTypeID	ENUM('D','R','T')	NOT NULL,
    PaymentDate		DATE				NOT NULL,
    PaymentAmount	FLOAT(5,1)			NOT NULL,
    PaymentComment	VARCHAR(30)			,
    FOREIGN KEY (InvoiceID) REFERENCES invoice(InvoiceID),
    FOREIGN KEY (PaymentTypeID) REFERENCES paymenttype(PaymentTypeID)
);

# paymenttype
CREATE TABLE paymenttype (
	PaymentTypeID		ENUM('D','R','T')	PRIMARY KEY,
    TypeDescription		VARCHAR(10)			NOT NULL
);

# invoiceline
CREATE TABLE invoiceline (
	InvoiceLineID		NUMERIC(4)		PRIMARY KEY,
    InvoiceID			NUMERIC(4)		NOT NULL,
    ProductID			NUMERIC(4)		NOT NULL,
    OrderedQuantity		FLOAT(3,1)		NOT NULL,
    FOREIGN KEY (InvoiceID) REFERENCES invoice(InvoiceID),
    FOREIGN KEY (ProductID) REFERENCES product(ProductID)
);

# product
CREATE TABLE product (
	ProductID				NUMERIC(4)		PRIMARY KEY,
    ProductLineID			NUMERIC(2)		NOT NULL,
    ProductDescription		VARCHAR(30)		,
    ProductFinish			VARCHAR(10)		,
    ProductStandardPrice	FLOAT(5,1)		NOT NULL,
    ProductOnHand			FLOAT(2,1)		NOT NULL,
    FOREIGN KEY (ProductLineID) REFERENCES productline(ProductLineID)
);

select * from product;

# productline
CREATE TABLE productline (
	ProductLineID		NUMERIC(2)		PRIMARY KEY,
    ProductLineName		VARCHAR(20)		NOT NULL
);

# uses
CREATE TABLE uses (
	MaterialID 			VARCHAR(20)		PRIMARY KEY,
    ProductID 			NUMERIC(4)		PRIMARY KEY,
    QuantityRequired	FLOAT(3,1)		NOT NULL,
    FOREIGN KEY (MaterialID) REFERENCES rawmaterial(MaterialID),
    FOREIGN KEY (ProductID) REFERENCES producT(ProductID)
);

# rawmaterial 
CREATE TABLE  rawmaterial (
	MaterialID				CHAR(12) 		PRIMARY KEY,
    MaterialName			VARCHAR(40)		NOT NULL,
    Thickness				VARCHAR(10)		,
    Width					SMALLINT(2)		,
    MatSize 				VARCHAR(30)		,
    Material 				VARCHAR(10)		,
    MaterialStandardPrice 	FLOAT(4,1)		NOT NULL,
    UnitOfMeasure 			VARCHAR(5)		NOT NULL,
    MaterialType 			VARCHAR(10)		NOT NULL
);

# supplies
CREATE TABLE supplies (
	VendorID 			NUMERIC(2)		PRIMARY KEY,
    MaterialID			VARCHAR(20)		PRIMARY KEY,
    SupplyUnitPrice		FLOAT(5,1)		NOT NULL,
    FOREIGN KEY (VendorID) REFERENCES vendor(VendorID)
);

# vendor
CREATE TABLE vendor (
	VendorID 			NUMERIC(2)		PRIMARY KEY,
    VendorName			VARCHAR(25)		NOT NULL,
    VendorAddress		VARCHAR(30)		NOT NULL,
    VendorCity			VARCHAR(20)		NOT NULL,
    VendorState			CHAR(2)			NOT NULL,
    VendorZipcode		VARCHAR(20)		NOT NULL,
    VendorPhone			CHAR(12)		NOT NULL,
    VendorFax			CHAR(12)		,
    VendorContact		VARCHAR(30)		,
    VendorTaxIdNumber	CHAR(9)			
);

select * from producedin;
