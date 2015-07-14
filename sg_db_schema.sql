--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: meter_data; Type: TABLE; Schema: public; Owner: sg_admin_db; Tablespace: 
--

CREATE TABLE meter_data (
    id_meter_data integer NOT NULL,
    id_meter integer NOT NULL,
    id_meter_data_collection integer NOT NULL
);


ALTER TABLE meter_data OWNER TO sg_admin_db;

--
-- Name: meter_data_collection; Type: TABLE; Schema: public; Owner: sg_admin_db; Tablespace: 
--

CREATE TABLE meter_data_collection (
    id_meter_data_collection integer NOT NULL,
    ini_date timestamp without time zone NOT NULL,
    end_date timestamp without time zone NOT NULL,
    min_val real,
    max_val real
);


ALTER TABLE meter_data_collection OWNER TO sg_admin_db;

--
-- Name: meter_data_collection_id_seq; Type: SEQUENCE; Schema: public; Owner: sg_admin_db
--

CREATE SEQUENCE meter_data_collection_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE meter_data_collection_id_seq OWNER TO sg_admin_db;

--
-- Name: meter_data_collection_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sg_admin_db
--

ALTER SEQUENCE meter_data_collection_id_seq OWNED BY meter_data_collection.id_meter_data_collection;


--
-- Name: meter_data_id_meter_data_seq; Type: SEQUENCE; Schema: public; Owner: sg_admin_db
--

CREATE SEQUENCE meter_data_id_meter_data_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE meter_data_id_meter_data_seq OWNER TO sg_admin_db;

--
-- Name: meter_data_id_meter_data_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sg_admin_db
--

ALTER SEQUENCE meter_data_id_meter_data_seq OWNED BY meter_data.id_meter_data;


--
-- Name: normalized_measure; Type: TABLE; Schema: public; Owner: sg_admin_db; Tablespace: 
--

CREATE TABLE normalized_measure (
    id_meter_data integer NOT NULL,
    normalized_measure real NOT NULL,
    date integer NOT NULL,
    id_normalized_measure integer NOT NULL
);


ALTER TABLE normalized_measure OWNER TO sg_admin_db;

--
-- Name: normalized_measure_id_normalized_measure_seq; Type: SEQUENCE; Schema: public; Owner: sg_admin_db
--

CREATE SEQUENCE normalized_measure_id_normalized_measure_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE normalized_measure_id_normalized_measure_seq OWNER TO sg_admin_db;

--
-- Name: normalized_measure_id_normalized_measure_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: sg_admin_db
--

ALTER SEQUENCE normalized_measure_id_normalized_measure_seq OWNED BY normalized_measure.id_normalized_measure;


--
-- Name: row_data; Type: TABLE; Schema: public; Owner: sg_admin_db; Tablespace: 
--

CREATE TABLE row_data (
    id_meter integer NOT NULL,
    date integer NOT NULL,
    kwh real
);


ALTER TABLE row_data OWNER TO sg_admin_db;

--
-- Name: id_meter_data; Type: DEFAULT; Schema: public; Owner: sg_admin_db
--

ALTER TABLE ONLY meter_data ALTER COLUMN id_meter_data SET DEFAULT nextval('meter_data_id_meter_data_seq'::regclass);


--
-- Name: id_meter_data_collection; Type: DEFAULT; Schema: public; Owner: sg_admin_db
--

ALTER TABLE ONLY meter_data_collection ALTER COLUMN id_meter_data_collection SET DEFAULT nextval('meter_data_collection_id_seq'::regclass);


--
-- Name: id_normalized_measure; Type: DEFAULT; Schema: public; Owner: sg_admin_db
--

ALTER TABLE ONLY normalized_measure ALTER COLUMN id_normalized_measure SET DEFAULT nextval('normalized_measure_id_normalized_measure_seq'::regclass);


--
-- Name: meter_data_pkey; Type: CONSTRAINT; Schema: public; Owner: sg_admin_db; Tablespace: 
--

ALTER TABLE ONLY meter_data
    ADD CONSTRAINT meter_data_pkey PRIMARY KEY (id_meter_data);


--
-- Name: normalized_measure_pkey; Type: CONSTRAINT; Schema: public; Owner: sg_admin_db; Tablespace: 
--

ALTER TABLE ONLY normalized_measure
    ADD CONSTRAINT normalized_measure_pkey PRIMARY KEY (id_normalized_measure);


--
-- Name: pk; Type: CONSTRAINT; Schema: public; Owner: sg_admin_db; Tablespace: 
--

ALTER TABLE ONLY meter_data_collection
    ADD CONSTRAINT pk PRIMARY KEY (id_meter_data_collection);


--
-- Name: row_data_pkey; Type: CONSTRAINT; Schema: public; Owner: sg_admin_db; Tablespace: 
--

ALTER TABLE ONLY row_data
    ADD CONSTRAINT row_data_pkey PRIMARY KEY (id_meter, date);


--
-- Name: normalized_measure_id_meter_data_date_idx; Type: INDEX; Schema: public; Owner: sg_admin_db; Tablespace: 
--

CREATE INDEX normalized_measure_id_meter_data_date_idx ON normalized_measure USING btree (id_meter_data, date);


--
-- Name: row_data_id_meter_date_idx; Type: INDEX; Schema: public; Owner: sg_admin_db; Tablespace: 
--

CREATE INDEX row_data_id_meter_date_idx ON row_data USING btree (id_meter, date);


--
-- Name: meter_data_id_meter_data_collection_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sg_admin_db
--

ALTER TABLE ONLY meter_data
    ADD CONSTRAINT meter_data_id_meter_data_collection_fkey FOREIGN KEY (id_meter_data_collection) REFERENCES meter_data_collection(id_meter_data_collection);


--
-- Name: normalized_measure_id_meter_data_fkey; Type: FK CONSTRAINT; Schema: public; Owner: sg_admin_db
--

ALTER TABLE ONLY normalized_measure
    ADD CONSTRAINT normalized_measure_id_meter_data_fkey FOREIGN KEY (id_meter_data) REFERENCES meter_data(id_meter_data);


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- Name: meter_data; Type: ACL; Schema: public; Owner: sg_admin_db
--

REVOKE ALL ON TABLE meter_data FROM PUBLIC;
REVOKE ALL ON TABLE meter_data FROM sg_admin_db;
GRANT ALL ON TABLE meter_data TO sg_admin_db;
GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE meter_data TO sg_user_db;


--
-- Name: meter_data_collection; Type: ACL; Schema: public; Owner: sg_admin_db
--

REVOKE ALL ON TABLE meter_data_collection FROM PUBLIC;
REVOKE ALL ON TABLE meter_data_collection FROM sg_admin_db;
GRANT ALL ON TABLE meter_data_collection TO sg_admin_db;
GRANT SELECT,INSERT,UPDATE ON TABLE meter_data_collection TO sg_user_db;


--
-- Name: meter_data_collection_id_seq; Type: ACL; Schema: public; Owner: sg_admin_db
--

REVOKE ALL ON SEQUENCE meter_data_collection_id_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE meter_data_collection_id_seq FROM sg_admin_db;
GRANT ALL ON SEQUENCE meter_data_collection_id_seq TO sg_admin_db;
GRANT SELECT,UPDATE ON SEQUENCE meter_data_collection_id_seq TO sg_user_db;


--
-- Name: meter_data_id_meter_data_seq; Type: ACL; Schema: public; Owner: sg_admin_db
--

REVOKE ALL ON SEQUENCE meter_data_id_meter_data_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE meter_data_id_meter_data_seq FROM sg_admin_db;
GRANT ALL ON SEQUENCE meter_data_id_meter_data_seq TO sg_admin_db;
GRANT SELECT,USAGE ON SEQUENCE meter_data_id_meter_data_seq TO sg_user_db;


--
-- Name: normalized_measure; Type: ACL; Schema: public; Owner: sg_admin_db
--

REVOKE ALL ON TABLE normalized_measure FROM PUBLIC;
REVOKE ALL ON TABLE normalized_measure FROM sg_admin_db;
GRANT ALL ON TABLE normalized_measure TO sg_admin_db;
GRANT SELECT,INSERT,DELETE ON TABLE normalized_measure TO sg_user_db;


--
-- Name: normalized_measure_id_normalized_measure_seq; Type: ACL; Schema: public; Owner: sg_admin_db
--

REVOKE ALL ON SEQUENCE normalized_measure_id_normalized_measure_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE normalized_measure_id_normalized_measure_seq FROM sg_admin_db;
GRANT ALL ON SEQUENCE normalized_measure_id_normalized_measure_seq TO sg_admin_db;
GRANT USAGE ON SEQUENCE normalized_measure_id_normalized_measure_seq TO sg_user_db;


--
-- Name: row_data; Type: ACL; Schema: public; Owner: sg_admin_db
--

REVOKE ALL ON TABLE row_data FROM PUBLIC;
REVOKE ALL ON TABLE row_data FROM sg_admin_db;
GRANT ALL ON TABLE row_data TO sg_admin_db;
GRANT SELECT ON TABLE row_data TO sg_user_db;


--
-- PostgreSQL database dump complete
--

