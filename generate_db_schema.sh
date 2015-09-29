#!/bin/bash
#su postgres -c "pg_dump -p 5423 -h localhost -s sg_db > /home/postgres/sg_db_schema.sql"
su postgres -c "pg_dump -p 5423 -h localhost -s sg_db" 
